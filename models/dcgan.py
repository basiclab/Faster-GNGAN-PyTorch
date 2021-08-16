import torch
import torch.nn as nn
import torch.nn.init as init

from .gradnorm import rescale_module


class Generator(nn.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    @torch.no_grad()
    def rescale_model(self, min_scale=1.0, max_scale=1.33):
        base_scale = 1.0
        for module in self.modules():
            base_scale, _ = rescale_module(module, base_scale)
        return base_scale

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=6)


class Discriminator32(Discriminator):
    def __init__(self, *args):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self, *args):
        super().__init__(M=48)


if __name__ == '__main__':
    Models = [
        (32, Discriminator32),
        (48, Discriminator48),
    ]
    for res, Model in Models:
        print("=" * 80)
        print(Model.__name__)
        x = torch.randn(2, 3, res, res, requires_grad=True).cuda()
        net_D = Model().cuda()
        f = net_D(x)
        grad_f = torch.autograd.grad(f.sum(), x)[0]
        grad_norm = torch.norm(torch.flatten(grad_f, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, 1)
        f_hat = f / (grad_norm + torch.abs(f))
        print('     '
              f'{"Output":>11s}, {"Raw Output":>11s}, {"Grad Norm":>11s}')
        print('ORIG '
              f'{f_hat[0].item():+11.7f}, '
              f'{f[0].item():+11.7f}, '
              f'{grad_norm[0].item():+11.7f}')

        for step in range(10):
            net_D.rescale_model()
            f_scaled = net_D(x)
            grad_f_scaled = torch.autograd.grad(f_scaled.sum(), x)[0]
            grad_norm_scaled = torch.norm(
                torch.flatten(grad_f_scaled, start_dim=1), p=2, dim=1)
            grad_norm_scaled = grad_norm_scaled.view(-1, 1)
            f_hat_scaled = f_scaled / (grad_norm_scaled + torch.abs(f_scaled))

            alpha1 = f / f_scaled
            alpha2 = grad_norm / grad_norm_scaled
            if step < 5:
                assert torch.allclose(
                    alpha1, alpha2, rtol=1e-04, atol=1e-06), \
                    f'{alpha1[0].item():.7f}, {alpha2[0].item():.7f}'
                assert torch.allclose(
                    f_hat, f_hat_scaled, rtol=1e-04, atol=1e-06), \
                    f'{f_hat[0].item():.7f}, {f_hat_scaled[0].item():.7f}'
            else:
                if not torch.allclose(alpha1, alpha2, rtol=1e-04, atol=1e-06):
                    print(f'WARN1 '
                          f'{alpha1[0].item():+.7f} != {alpha2[0].item():+.7f}')
                if not torch.allclose(
                        f_hat, f_hat_scaled, rtol=1e-04, atol=1e-06):
                    print(f'WARN2 '
                          f'{f_hat[0].item():+.7f}, '
                          f'{f_hat_scaled[0].item():+.7f}')

            print('PASS '
                  f'{f_hat_scaled[0].item():+11.7f}, '
                  f'{f_scaled[0].item():+11.7f}, '
                  f'{grad_norm_scaled[0].item():+11.7f}')