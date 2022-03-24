import torch
import torch.nn as nn
import torch.nn.init as init

from .gradnorm import Rescalable


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
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)
        self.blocks = nn.Sequential(
            GenBlock(256, 256),
            GenBlock(256, 256),
            GenBlock(256, 256),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        x = self.output(self.blocks(z))
        return (x + 1) / 2


class ResGenerator48(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(z_dim, 6 * 6 * 512)
        self.blocks = nn.Sequential(
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        z = self.linear(z)
        z = z.view(-1, 512, 6, 6)
        x = self.output(self.blocks(z))
        return (x + 1) / 2


class ResGenerator128(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 1024)

        self.blocks = nn.Sequential(
            GenBlock(1024, 1024),
            GenBlock(1024, 512),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        z = self.linear(z)
        z = z.view(-1, 1024, 4, 4)
        x = self.output(self.blocks(z))
        return (x + 1) / 2


class ResGenerator256(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 1024)

        self.blocks = nn.Sequential(
            GenBlock(1024, 1024),
            GenBlock(1024, 512),
            GenBlock(512, 512),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        z = self.linear(z)
        z = z.view(-1, 1024, 4, 4)
        x = self.output(self.blocks(z))
        return (x + 1) / 2


class ReScaleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.shortcut_scale = 1

    @torch.no_grad()
    def rescale_block(self, base_scale, alpha=1.):
        residual_scale = base_scale
        for module in self.residual.modules():
            if isinstance(module, Rescalable):
                residual_scale = module.rescale(residual_scale, alpha)

        shortcut_scale = base_scale
        for module in self.shortcut.modules():
            if isinstance(module, Rescalable):
                shortcut_scale = module.rescale(shortcut_scale, alpha)
        self.shortcut_scale = residual_scale / shortcut_scale

        return residual_scale

    def forward(self, x):
        return self.residual(x) + self.shortcut(x) * self.shortcut_scale


class OptimizedDisblock(ReScaleBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            Rescalable(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        # residual
        self.residual = nn.Sequential(
            Rescalable(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(True),
            Rescalable(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, Rescalable):
                init.kaiming_normal_(m.module.weight)
                init.zeros_(m.module.bias)
                m.init_module_scale()


class DisBlock(ReScaleBlock):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                Rescalable(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # residual
        residual = [
            nn.ReLU(),
            Rescalable(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(True),
            Rescalable(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, Rescalable):
                init.kaiming_normal_(m.module.weight)
                init.zeros_(m.module.bias)
                m.init_module_scale()


class ReScaleModel(nn.Module):
    def rescale_model(self, alpha=1.):
        base_scale = 1
        for block in self.model:
            if isinstance(block, ReScaleBlock):
                base_scale = block.rescale_block(base_scale)
        base_scale = self.linear.rescale(base_scale, alpha)
        return base_scale

    def forward(self, x, *args, **kwargs):
        x = x * 2 - 1
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

    def initialize(self):
        init.kaiming_normal_(self.linear.module.weight)
        init.zeros_(self.linear.module.bias)
        self.linear.init_module_scale()


class ResDiscriminator32(ReScaleModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 128),
            DisBlock(128, 128, down=True),
            DisBlock(128, 128),
            DisBlock(128, 128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = Rescalable(nn.Linear(128, 1))
        # initialize weight
        self.initialize()


class ResDiscriminator48(ReScaleModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = Rescalable(nn.Linear(512, 1))
        # initialize weight
        self.initialize()


class ResDiscriminator128(ReScaleModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            DisBlock(512, 1024, down=True),
            DisBlock(1024, 1024),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = Rescalable(nn.Linear(1024, 1))
        # initialize weight
        self.initialize()


class ResDiscriminator256(ReScaleModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            DisBlock(512, 512, down=True),
            DisBlock(512, 1024, down=True),
            DisBlock(1024, 1024),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = Rescalable(nn.Linear(1024, 1))
        # initialize weight
        self.initialize()
