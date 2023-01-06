import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import sngan, dcgan
from training.datasets import Dataset
from vis.core import style, calc_grad_norm


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs/Ablation"
runs = {
    "SN-9L": ("SN_cifar10_cnn9", sngan.Discriminator9, dcgan.Generator),
    "SN-6L": ("SN_cifar10_cnn6", sngan.Discriminator6, dcgan.Generator),
    "SN-3L": ("SN_cifar10_cnn3", sngan.Discriminator3, dcgan.Generator),
}
markers = ['o', 'D', '^']


class SubNetwork(torch.nn.Module):
    def __init__(self, num_layers, model):
        super().__init__()
        self.num_layers = num_layers
        self.model = model

    def forward(self, x):
        counter = 0
        for module in self.model.modules():
            if isinstance(module, (
                    sngan.SpectralNorm,
                    torch.nn.LeakyReLU,
                    torch.nn.Flatten
            )):
                x = module(x)
                if isinstance(module, sngan.SpectralNorm):
                    counter += 1
                    if counter == self.num_layers:
                        break
        return x


@torch.no_grad()
def estimate_max_gn(D_all, G, loader, z_dim=128):
    G.eval()
    D_all.eval()
    D_all.requires_grad_(False)

    total_layers = 0
    for module in D_all.modules():
        if isinstance(module, sngan.SpectralNorm):
            total_layers += 1

    max_grad_norms = []
    steps = []
    for num_layers in range(1, total_layers + 1):
        D = SubNetwork(num_layers, D_all)
        max_grad_norm = 0
        with tqdm(loader, ncols=0, desc=f"{num_layers}", leave=False) as pbar:
            for real, _, _ in pbar:
                real = real.to(device)
                fake = G(torch.randn(real.size(0), z_dim, device=device))
                midd = torch.lerp(real, fake, torch.rand(real.size(0), 1, 1, 1, device=device))
                # rand = torch.rand_like(real) * 2 - 1

                # real data points
                init_gn, last_fn, max_gn = calc_grad_norm(D, real)
                max_grad_norm = max(max_grad_norm, max_gn)
                pbar.write(f"{init_gn:.4f} -> {last_fn:.4f}: max is {max_gn:.4f}")
                # fake data points
                # init_gn, last_fn, max_gn = calc_grad_norm(D, fake)
                # max_grad_norm = max(max_grad_norm, max_gn)
                # interpolation points
                # init_gn, last_fn, max_gn = calc_grad_norm(D, midd)
                # max_grad_norm = max(max_grad_norm, max_gn)
                # random points
                # init_gn, last_fn, max_gn = calc_grad_norm(D, rand)
                # max_grad_norm = max(max_grad_norm, max_gn)

                pbar.set_postfix(max_grad_norm=max_grad_norm)

        max_grad_norms.append(max_grad_norm)
        steps.append(num_layers)

    assert torch.allclose(D_all(real), D(real))

    return steps, max_grad_norms


def main():
    os.makedirs(cache_dir, exist_ok=True)

    indices = torch.randperm(
        50000, generator=torch.Generator().manual_seed(0))[:5000]
    dataset = torch.utils.data.Subset(
        Dataset('./data/cifar10', hflip=False, resolution=32, cr=False),
        indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4, shuffle=False)

    plot_lipschitz_data = dict()
    with tqdm(runs.items(), ncols=0, leave=False) as pbar:
        for legend, (log_name, MODEL_D, MODEL_G) in pbar:
            pbar.set_description(legend)
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_decaylip.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {legend}_{seed} from cache")
                plot_lipschitz_data[legend] = torch.load(cache_path)
                continue

            D = MODEL_D(resolution=32, n_classes=None).to(device)
            G = MODEL_G(resolution=32, n_classes=None, z_dim=128).to(device)
            D(torch.rand((1, 3, 32, 32)).to(device))
            ckpt = torch.load(os.path.join(log_path, '200000.pt'))
            G.load_state_dict(ckpt['G'])
            D.load_state_dict(ckpt['D'])

            steps, max_grad_norms = estimate_max_gn(D, G, loader)
            plot_lipschitz_data[legend] = (steps, max_grad_norms)
            pbar.write(", ".join(f"{x:.3f}" for x in max_grad_norms))
            torch.save(plot_lipschitz_data[legend], cache_path)

    # ============================= plot =============================

    plt.figure(figsize=(8, 7))
    lines = []
    legends = []
    for marker, (legend, (x, max_grad_norms)) in zip(
            markers, plot_lipschitz_data.items()):
        x = np.array(x)
        y = np.array(max_grad_norms)
        y = np.flip(np.maximum.accumulate(np.flip(y)))

        marker, = plt.plot(x, y, linestyle='', marker=marker)
        line2d, = plt.plot(x, y, alpha=0.8, color=marker.get_color())

        lines.append((marker, line2d))
        legends.append(legend)

    xticks = range(1, 11)
    plt.xticks(xticks)
    plt.xlabel('$k$')

    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(yticks)
    plt.ylabel(r'$\max_x\Vert\nabla_x f_k(x)\Vert$')
    plt.grid(axis='x')

    plt.legend(
        lines, legends, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_decaylip.png'))
    print("Saved to", os.path.join(save_dir, 'vis_decaylip.png'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    with style():
        main()
