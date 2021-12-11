import torch
import torch.nn as nn
import torch.nn.init as init

from .gradnorm import Rescalable


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
            Rescalable(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            Rescalable(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            Rescalable(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            Rescalable(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            Rescalable(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            Rescalable(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            Rescalable(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = Rescalable(nn.Linear(M // 8 * M // 8 * 512, 1))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, Rescalable):
                init.normal_(m.module.weight, std=0.02)
                init.zeros_(m.module.bias)
                m.init_module_scale()

    def rescale_model(self, alpha=1.):
        base_scale = 1.0
        for module in self.modules():
            if isinstance(module, Rescalable):
                base_scale = module.rescale(base_scale, alpha)
        return base_scale

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__(z_dim, M=6)


class Discriminator32(Discriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(M=48)
