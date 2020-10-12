import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channel, n_classes):
        super().__init__()
        self.gain = nn.Embedding(n_classes, in_channel)
        self.bias = nn.Embedding(n_classes, in_channel)
        self.register_buffer('stored_mean', torch.zeros(in_channel))
        self.register_buffer('stored_var',  torch.ones(in_channel))

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1) + 1
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = F.batch_norm(
            x, self.stored_mean, self.stored_var, None, None, self.training)
        return x * gain + bias


class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()

        # residual
        self.bn1 = ConditionalBatchNorm2d(in_channels, n_classes)
        self.residual1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.bn2 = ConditionalBatchNorm2d(out_channels, n_classes)
        self.residual2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0))

    def forward(self, x, y):
        h = self.residual1(self.bn1(x, y))
        h = self.residual2(self.bn2(h, y))
        return h + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, n_classes, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = nn.ModuleList([
            ResGenBlock(256, 256, n_classes),
            ResGenBlock(256, 256, n_classes),
            ResGenBlock(256, 256, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class ResGenerator128(nn.Module):
    def __init__(self, n_classes, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 1024)

        self.blocks = nn.ModuleList([
            ResGenBlock(1024, 1024, n_classes),
            ResGenBlock(1024, 512, n_classes),
            ResGenBlock(512, 256, n_classes),
            ResGenBlock(256, 128, n_classes),
            ResGenBlock(128, 64, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 1024, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = spectral_norm(nn.Linear(128, 1, bias=False))
        self.embedding = spectral_norm(nn.Embedding(n_classes, 128))
        weights_init(self)

    def forward(self, x, y):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x) + (self.embedding(y) * x).sum(dim=1, keepdim=True)
        return x


class ResDiscriminator128(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 64),
            ResDisBlock(64, 128, down=True),
            ResDisBlock(128, 256, down=True),
            ResDisBlock(256, 512, down=True),
            ResDisBlock(512, 1024, down=True),
            nn.ReLU())
        self.linear = spectral_norm(nn.Linear(1024, 1))
        self.embedding = spectral_norm(nn.Embedding(n_classes, 1024))
        weights_init(self)

    def forward(self, x, y):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x) + (self.embedding(y) * x).sum(dim=1, keepdim=True)
        return x


class GenDis(nn.Module):
    """
    As suggest in official PyTorch implementation, paralleling generator and
    discriminator together can avoid gathering fake images in the
    intermediate stage
    """
    def __init__(self, net_G, net_D):
        super().__init__()
        self.net_G = net_G
        self.net_D = net_D

    def forward(self, z, y_fake, x_real=None, y_real=None, **kwargs):
        if x_real is not None and y_real is not None:
            with torch.no_grad():
                x_fake = self.net_G(z, y_fake)
            x = torch.cat([x_real, x_fake], dim=0)
            y = torch.cat([y_real, y_fake], dim=0)
            pred = self.net_D(x, y=y)
            net_D_real, net_D_fake = torch.split(
                pred, [x_real.shape[0], x_fake.shape[0]])
            return net_D_real, net_D_fake
        else:
            x_fake = self.net_G(z, y_fake)
            net_D_fake = self.net_D(x_fake, y=y_fake)
            return net_D_fake


def weights_init(m):
    modules = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)
    for name, module in m.named_modules():
        if isinstance(module, modules):
            if 'residual' in name:
                torch.nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


generators = {
    'res32': ResGenerator32,
    'res128': ResGenerator128,
}

discriminators = {
    'res32': ResDiscriminator32,
    'res128': ResDiscriminator128,
}
