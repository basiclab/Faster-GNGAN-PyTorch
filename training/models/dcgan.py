import torch
import torch.nn as nn

from .base import Reshape, RescalableWrapper, RescalableSequentialModel


class Generator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__()
        config = {
            32: 4,
            48: 6,
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        M = config[resolution]

        self.main = nn.Sequential(
            nn.Linear(z_dim, M * M * 512),
            Reshape(-1, 512, M, M),
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

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.main(z)
        return x


class Discriminator(RescalableSequentialModel):
    def __init__(self, resolution, n_classes):
        super().__init__()
        config = {
            32: 4,
            48: 6,
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        M = config[resolution]

        self.main = nn.Sequential(
            # H x W
            RescalableWrapper(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            RescalableWrapper(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # H / 2 x W / 2
            RescalableWrapper(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            RescalableWrapper(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # H / 4 x W / 4
            RescalableWrapper(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            RescalableWrapper(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # H / 8 x W / 8
            RescalableWrapper(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(start_dim=1),
            RescalableWrapper(nn.Linear(M * M * 512, 1)),
        )

        for m in self.modules():
            if isinstance(m, RescalableWrapper):
                torch.nn.init.normal_(m.module.weight, std=0.02)
                torch.nn.init.zeros_(m.module.bias)
                m.init_module()

    def forward(self, x, *args, **kwargs):
        y = self.main(x)
        return y
