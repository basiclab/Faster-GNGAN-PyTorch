import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from torch.utils.data import DataLoader, Dataset

from training.gn import normalize_D, vanilla_D
from training.losses import ns_loss_D, bce_loss_D
from training.misc import set_seed
from vis.core import style


device = torch.device('cuda:0')


class OptimalDiscriminator(nn.Module):
    def __init__(self, C1, C2, std):
        super().__init__()
        self.register_buffer('C1', torch.tensor(C1))
        self.register_buffer('C2', torch.tensor(C2))
        self.register_buffer('std', torch.tensor(std))
        self.register_buffer('pi', torch.tensor(np.pi))

    def p(self, x, mu, std):
        """Probability of multivariate Gaussian"""
        num = torch.exp((-1 / (2 * std)) * ((x - mu) * (x - mu)).sum(dim=1))
        den = (2 * self.pi * std)
        return num / den

    def forward(self, x):
        p_real = self.p(x, self.C1, self.std)
        p_fake = self.p(x, self.C2, self.std)
        p = p_real / (p_real + p_fake)
        p = p.unsqueeze(-1)
        return p
        # logits = torch.log(p / (1 - p + 1e-10))
        # return logits.unsqueeze(-1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


class GaussianDataset(Dataset):
    def __init__(self, N, C1, C2, std):
        self.N = N
        self.C1 = torch.tensor(C1)
        self.C2 = torch.tensor(C2)
        self.x = torch.randn(N, 2) * std + self.C1
        self.std = std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        real = self.x[idx]                # real data
        fake = torch.randn(2) * self.std + self.C2   # fake data
        return real, fake


@torch.no_grad()
def eval_acc(D, dataloader):
    true_n = 0
    total_n = 0
    for real, fake in dataloader:
        real = real.to(device)
        fake = fake.to(device)
        y_real = D(real)
        y_fake = D(fake)
        true_n += (y_real > 0).sum().item()
        true_n += (y_fake < 0).sum().item()
        total_n += y_real.numel() + y_fake.numel()
    return true_n / total_n


def fit(D, dataloader, optimizer, epochs):
    D.train()
    acc = eval_acc(D, dataloader)
    print(f"Epoch: {0:3d}, Accuracy: {acc:.3f}")
    for epoch in range(epochs):
        for real, fake in dataloader:
            real = real.to(device)
            fake = fake.to(device)
            x = torch.cat([real, fake], dim=0)
            loss_real, loss_fake = ns_loss_D(D(x))
            loss = loss_real.mean() + loss_fake.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = eval_acc(D, dataloader)
        print(f"Epoch: {epoch + 1:3d}, Accuracy: {acc:.3f}")


def fit_gp(D, dataloader, optimizer, epochs):
    D.train()
    acc = eval_acc(D, dataloader)
    print(f"Epoch: {0:3d}, Accuracy: {acc:.3f}")
    for epoch in range(epochs):
        for real, fake in dataloader:
            real = real.to(device)
            fake = fake.to(device)
            x = torch.cat([real, fake], dim=0)
            loss_real, loss_fake = ns_loss_D(D(x))
            loss_gan = loss_real.mean() + loss_fake.mean()

            # GP
            # t = torch.rand(real.shape[0], 1, 1, 1, device=device)
            # x = t * real + (1 - t) * fake
            x.requires_grad_()
            y = D(x)
            grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            grad_norm = torch.linalg.vector_norm(grad.flatten(start_dim=1), dim=1)
            loss_gp = ((grad_norm - 1) ** 2).mean()

            loss = loss_gan + 10 * loss_gp
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = eval_acc(D, dataloader)
        print(f"Epoch: {epoch + 1:3d}, Accuracy: {acc:.3f}")


def fit_gn(D, dataloader, optimizer, epochs):
    D.train()
    acc = eval_acc(D, dataloader)
    print(f"Epoch: {0:3d}, Accuracy: {acc:.3f}")
    for epoch in range(epochs):
        for real, fake in dataloader:
            real = real.to(device)
            fake = fake.to(device)
            x = torch.cat([real, fake], dim=0)
            _, (loss_real, loss_fake), _ = normalize_D(D, x, ns_loss_D)
            loss = loss_real.mean() + loss_fake.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = eval_acc(D, dataloader)
        print(f"Epoch: {epoch + 1:3d}, Accuracy: {acc:.3f}")


def plot(D, x1, x2, normalize_fn, name):
    x1s, x2s = torch.meshgrid(x1, x2, indexing='xy')
    x = torch.stack([x1s.flatten(), x2s.flatten()], dim=1)

    D.eval()
    grad_norms = []
    decisions = []
    for batch_x in x.split(64):
        batch_x = batch_x.to(device)
        batch_x.requires_grad_(True)
        y, _, _ = normalize_fn(D, batch_x, ns_loss_D)
        if name == 'gn':
            y = (y + 1) / 2
            # y = y.sigmoid()
        if name in ['gp', 'gan']:
            y = y.sigmoid()
        grad = torch.autograd.grad(y.sum(), batch_x)[0]
        grad_norm = torch.linalg.vector_norm(grad.flatten(start_dim=1), dim=1)
        grad_norms.append(grad_norm.cpu())
        decisions.append(y.detach().cpu())
    grad_norms = torch.cat(grad_norms, dim=0)
    decisions = torch.cat(decisions, dim=0)

    # prepare data
    X = x1s.numpy()
    Y = x2s.numpy()
    Z = grad_norms.numpy().reshape(X.shape)
    D = decisions.numpy().reshape(X.shape)

    # ======================== decision ========================

    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(
        X, Y, D, cmap=cm.coolwarm, linewidth=0, antialiased=False,
        vmin=0, vmax=1)

    ax.zaxis._axinfo['juggled'] = (1, 2, 0)     # zaxis on left
    ax.zaxis.set_rotate_label(False)            # disable automatic rotation
    ax.set_zlabel('Porbability', labelpad=20, rotation=90)
    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])

    ax.set_xlim(0, 7)
    ax.set_xlabel('$x_1$', labelpad=15)

    ax.set_ylim(0, 7)
    ax.set_ylabel('$x_2$', labelpad=15)

    plt.colorbar(surf, ticks=[0.0, 0.5, 1.0], shrink=0.7, aspect=10, anchor=(-0.5, 0.5))
    plt.tight_layout(rect=[0.05, 0.0, 1.1, 1.05])
    plt.savefig(f'surface_dec_{name}.png')
    plt.clf()

    # ======================== grad norm ========================

    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False,
        vmin=0, vmax=max(1, np.max(Z)))

    ax.zaxis.set_rotate_label(False)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)
    ax.set_zlabel(r'$\Vert\nabla_x\hat{D}(x)\Vert$', labelpad=20, rotation=90)
    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])

    ax.set_xlim(0, 7)
    ax.set_xlabel('$x_1$', labelpad=15)

    ax.set_ylim(0, 7)
    ax.set_ylabel('$x_2$', labelpad=15)

    plt.colorbar(surf, ticks=[0.0, 0.5, 1.0], shrink=0.7, aspect=10, anchor=(-0.5, 0.5))
    plt.tight_layout(rect=[0.05, 0.0, 1.1, 1.05])
    plt.savefig(f'surface_grad_{name}.png')
    plt.clf()


def main():
    set_seed(0)
    C1 = [2.5, 3.5]     # real
    C2 = [4.5, 3.5]     # fake
    std = np.sqrt(2)
    dataset = GaussianDataset(1000, C1, C2, std)
    x1 = torch.linspace(0, 7, 200)
    x2 = torch.linspace(0, 7, 200)

    D_gan = Discriminator().to(device)
    D_gp = Discriminator().to(device)
    D_gn = Discriminator().to(device)
    D_opt = OptimalDiscriminator(C1, C2, std).to(device)
    D_gp.load_state_dict(D_gan.state_dict())
    D_gn.load_state_dict(D_gan.state_dict())

    optim_gan = torch.optim.Adam(D_gan.parameters(), lr=1e-3, betas=(0.0, 0.9))
    optim_gp = torch.optim.Adam(D_gp.parameters(), lr=1e-3, betas=(0.0, 0.9))
    optim_gn = torch.optim.Adam(D_gn.parameters(), lr=1e-3, betas=(0.0, 0.9))

    print("Fit GAN")
    fit(D_gan, DataLoader(dataset, batch_size=32), optim_gan, epochs=10)
    print("Fit WGAN-GP")
    fit_gp(D_gp, DataLoader(dataset, batch_size=32), optim_gp, epochs=10)
    print("Fit GN-GAN")
    fit_gn(D_gn, DataLoader(dataset, batch_size=32), optim_gn, epochs=10)

    plot(D_gan, x1, x2, vanilla_D, 'gan')
    plot(D_opt, x1, x2, vanilla_D, 'opt')
    plot(D_gp, x1, x2, vanilla_D, 'gp')
    plot(D_gn, x1, x2, normalize_D, 'gn')


if __name__ == '__main__':
    with style(**{'axes.grid': False}):
        main()
