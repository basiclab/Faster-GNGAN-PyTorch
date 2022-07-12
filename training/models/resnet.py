import torch
import torch.nn as nn
import torch.nn.init as init

from .base import Reshape, RescalableSequentialModel, RescalableResBlock, RescalableWrapper


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        # main
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__()
        config = {
            32: [256, 256, 256, 256],
            48: [512, 256, 128, 64],
            128: [1024, 1024, 512, 256, 128, 64],
            256: [1024, 1024, 512, 512, 256, 128, 64],
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        channels = config[resolution]
        init_resolution = resolution // (2 ** (len(channels) - 1))

        # layers
        blocks = [
            nn.Linear(z_dim, init_resolution * init_resolution * channels[0]),
            Reshape(-1, channels[0], init_resolution, init_resolution)
        ]
        for i in range(1, len(channels)):
            blocks.append(GenBlock(channels[i - 1], channels[i]))
        blocks.extend([
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(True),
            nn.Conv2d(channels[-1], 3, 3, stride=1, padding=1),
            nn.Tanh(),
        ])
        self.main = nn.Sequential(*blocks)

        # initialize weight
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.main(z)
        return x


class FirstDisBlock(RescalableResBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        # main
        self.main = nn.Sequential(
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(True),
            RescalableWrapper(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2))


class DisBlock(RescalableResBlock):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                RescalableWrapper(
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # main
        residual = [
            nn.ReLU(),
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(True),
            RescalableWrapper(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.main = nn.Sequential(*residual)


class Discriminator(RescalableSequentialModel):
    def __init__(self, resolution, n_classes):
        super().__init__()
        config = {
            # channels, down_layers
            32: ([128, 128, 128, 128], 2),
            48: ([64, 128, 256, 512], 4),
            128: ([64, 128, 256, 512, 1024, 1024], 5),
            256: ([64, 128, 256, 512, 512, 1024, 1024], 6),
        }
        assert resolution in config, "The resolution %d is not supported in Discriminator." % resolution
        channels, down_layers = config[resolution]

        blocks = [FirstDisBlock(3, channels[0])]
        for i in range(1, len(channels)):
            blocks.append(
                DisBlock(channels[i - 1], channels[i], down=i < down_layers))
        blocks.extend([
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            RescalableWrapper(nn.Linear(channels[-1], 1)),
        ])
        self.main = nn.Sequential(*blocks)

        # initialize weight
        for m in self.modules():
            if isinstance(m, RescalableWrapper):
                init.kaiming_normal_(m.module.weight)
                init.zeros_(m.module.bias)
                m.init_module()

    def forward(self, x, *args, **kwargs):
        y = self.main(x)
        return y
