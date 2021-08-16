import torch
import torch.nn as nn
import torch.nn.init as init

from .gradnorm import rescale_module


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
    def __init__(self, z_dim, *args):
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
        return self.output(self.blocks(z))


class ResGenerator48(nn.Module):
    def __init__(self, z_dim, *args):
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
        return self.output(self.blocks(z))


class ResGenerator128(nn.Module):
    def __init__(self, z_dim):
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

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 1024, 4, 4)
        return self.output(self.blocks(inputs))


class ResGenerator256(nn.Module):
    def __init__(self, z_dim):
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

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 1024, 4, 4)
        return self.output(self.blocks(inputs))


class ReScaleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.shortcut_scale = 1

    @torch.no_grad()
    def rescale_block(self, base_scale, min_scale=0.7, max_scale=1.0):
        residual_scale = base_scale
        for module in self.residual.modules():
            residual_scale, _ = rescale_module(module, residual_scale)

        shortcut_scale = base_scale
        for module in self.shortcut.modules():
            shortcut_scale, _ = rescale_module(module, shortcut_scale)
        self.shortcut_scale *= residual_scale / shortcut_scale

        return residual_scale

    def forward(self, x):
        return self.residual(x) + self.shortcut(x) * self.shortcut_scale


class OptimizedDisblock(ReScaleBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)


class DisBlock(ReScaleBlock):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # residual
        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)


class ReScaleModel(nn.Module):
    def rescale_model(self, min_scale=0.7, max_scale=1.0):
        base_scale = 1
        for block in self.model:
            if isinstance(block, ReScaleBlock):
                base_scale = block.rescale_block(base_scale)
        base_scale, _ = rescale_module(self.linear, base_scale)
        return base_scale

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class ResDiscriminator32(ReScaleModel):
    def __init__(self, *args):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 128),
            DisBlock(128, 128, down=True),
            DisBlock(128, 128),
            DisBlock(128, 128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(128, 1)
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)


class ResDiscriminator48(ReScaleModel):
    def __init__(self, *args):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(512, 1)
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)


class ResDiscriminator128(ReScaleModel):
    def __init__(self):
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
        self.linear = nn.Linear(1024, 1)
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)


class ResDiscriminator256(ReScaleModel):
    def __init__(self):
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
        self.linear = nn.Linear(1024, 1)
        # initialize weight
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)


if __name__ == '__main__':
    Models = [
        (32, ResDiscriminator32),
        (48, ResDiscriminator48),
        (128, ResDiscriminator128),
        (256, ResDiscriminator256),
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