import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .spectral_norm import spectral_norm


class Generator(nn.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


# In CR-SNGAN, the channel sizes are    [64, 128, 128, 256, 256, 512, 512]
# In SNGAN, the channel sizes are       [64, 64, 128, 128, 256, 256, 512]
class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            spectral_norm(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                dim=(3, M, M)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                dim=(64, M, M)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            spectral_norm(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                dim=(128, M // 2, M // 2)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                dim=(128, M // 2, M // 2)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            spectral_norm(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                dim=(256, M // 4, M // 4)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                dim=(256, M // 4, M // 4)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            spectral_norm(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                dim=(512, M // 8, M // 8)),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = spectral_norm(
            nn.Linear(M // 8 * M // 8 * 512, 1), dim=(M // 8 * M // 8 * 512,))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=6)


class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self):
        super().__init__(M=48)


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        # residual
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)
        self.blocks = nn.Sequential(
            GenBlock(256, 256),
            GenBlock(256, 256),
            GenBlock(256, 256),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        return self.output(self.blocks(inputs))


class ResGenerator48(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 6 * 6 * 512)
        self.blocks = nn.Sequential(
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 512, 6, 6)
        return self.output(self.blocks(inputs))


class OptimizedDisblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                size=(in_channels, size, size)))
        # residual
        self.residual = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                size=(in_channels, size, size)),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                size=(out_channels, size, size)),
            nn.AvgPool2d(2))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) * 4 + self.shortcut(F.avg_pool2d(x, 2) * 4)


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, down=False):
        super().__init__()
        self.down = down
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                dim=(in_channels, size, size)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # residual
        residual = [
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                dim=(in_channels, size, size)),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                dim=(out_channels, size, size)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        if self.down:
            return (self.residual(x) + self.shortcut(x)) * 4
        else:
            return self.residual(x) + self.shortcut(x)


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 128),
            DisBlock(128, 128, down=True),
            DisBlock(128, 128),
            DisBlock(128, 128),
            nn.ReLU(inplace=True))
        self.linear = spectral_norm(nn.Linear(128, 1, bias=False), dim=(128,))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class ResDiscriminator48(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            nn.ReLU())
        self.linear = spectral_norm(nn.Linear(512, 1, bias=False), dim=(512,))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class GenDis(nn.Module):
    """
    Speed up training by paralleling generator and discriminator together
    """
    def __init__(self, net_G, net_D):
        super().__init__()
        self.net_G = net_G
        self.net_D = net_D

    def forward(self, z, real=None):
        if real is not None:
            with torch.no_grad():
                fake = self.net_G(z).detach()
            x = torch.cat([real, fake], dim=0)
            pred = self.net_D(x)
            net_D_real, net_D_fake = torch.split(
                pred, [real.shape[0], fake.shape[0]])
            return net_D_real, net_D_fake
        else:
            fake = self.net_G(z)
            net_D_fake = self.net_D(fake)
            return net_D_fake
