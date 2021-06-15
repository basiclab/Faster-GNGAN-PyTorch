import math

import torch
import torch.nn as nn
import torch.nn.init as init


def normalize(x):
    norm = torch.norm(torch.flatten(x))
    return x / (norm + 1e-12)


class SpectralNorm(nn.Module):
    def __init__(self, module, n_iteration=10):
        super().__init__()
        self.module = module
        self.n_iteration = n_iteration
        self.register_buffer('u', None)
        self.register_buffer('v', None)

    def forward(self, x):
        # weight = self.module.weight.view(self.module.weight.size(0), -1)
        if self.u is None:
            initialize_forward = True
            self.v = normalize(torch.randn_like(x[0])).unsqueeze(0)
            with torch.no_grad():
                self.u = normalize(self.forward_nobias(self.v))
        else:
            initialize_forward = False
        if self.training:
            with torch.no_grad():
                u = self.u
                if initialize_forward:
                    n_iteration = self.n_iteration * 10
                else:
                    n_iteration = self.n_iteration
                for _ in range(n_iteration):
                    v = normalize(self.backward_nobias(u))
                    u = normalize(self.forward_nobias(v))
                if self.n_iteration > 0:
                    self.u = u.clone(memory_format=torch.contiguous_format)
                    self.v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.sum(self.v * self.backward_nobias(self.u))
        return self.forward_nobias(
            x, self.module.weight / sigma, self.module.bias)

    def forward_nobias(self, x, weight=None, bias=None):
        if weight is None:
            weight = self.module.weight
        if isinstance(self.module, nn.Conv2d):
            x_ou = torch.nn.functional.conv2d(
                x, weight, bias=bias, stride=self.module.stride,
                padding=self.module.padding)
        if isinstance(self.module, nn.Linear):
            x_ou = torch.nn.functional.linear(
                x, weight, bias=bias)
        return x_ou

    def backward_nobias(self, x, weight=None, bias=None):
        if weight is None:
            weight = self.module.weight
        if isinstance(self.module, nn.Conv2d):
            x_ou = torch.nn.functional.conv_transpose2d(
                x, weight, bias=bias, stride=self.module.stride,
                padding=self.module.padding)
        if isinstance(self.module, nn.Linear):
            x_ou = torch.nn.functional.linear(
                x, weight.T, bias=bias)
        return x_ou


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

    def forward(self, z):
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
            SpectralNorm(nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            SpectralNorm(nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            SpectralNorm(nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(
                256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            SpectralNorm(nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = SpectralNorm(nn.Linear(M // 8 * M // 8 * 512, 1))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, SpectralNorm):
                if isinstance(m.module, (nn.Conv2d, nn.Linear)):
                    init.normal_(m.module.weight, std=0.02)
                    init.zeros_(m.module.bias)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=4)


class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__(M=32)


class DiscriminatorK(nn.Module):
    def __init__(self, num_per_block=1, M=32):
        super().__init__()
        blocks = []
        now_ch = 3
        for ch in [64, 128, 256]:
            for _ in range(num_per_block - 1):
                blocks.append(SpectralNorm(nn.Conv2d(
                    now_ch, ch, kernel_size=3, stride=1, padding=1)))
                blocks.append(nn.LeakyReLU(0.1, inplace=True))
                now_ch = ch
            blocks.append(SpectralNorm(nn.Conv2d(
                now_ch, ch * 2, kernel_size=4, stride=2, padding=1)))
            blocks.append(nn.LeakyReLU(0.1, inplace=True))
            now_ch = ch * 2
        self.main = nn.Sequential(*blocks)

        self.linear = SpectralNorm(nn.Linear(M // 8 * M // 8 * 512, 1))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, SpectralNorm):
                if isinstance(m.module, (nn.Conv2d, nn.Linear)):
                    init.normal_(m.module.weight, std=0.02)
                    init.zeros_(m.module.bias)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Discriminator3(DiscriminatorK):
    def __init__(self):
        super().__init__(num_per_block=1)


class Discriminator6(DiscriminatorK):
    def __init__(self):
        super().__init__(num_per_block=2)


class Discriminator9(DiscriminatorK):
    def __init__(self):
        super().__init__(num_per_block=3)


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


class Scalar(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return x * self.scalar


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            Scalar(4),
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        # residual
        self.residual = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
            nn.AvgPool2d(2),
            Scalar(4))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, SpectralNorm) and isinstance(m.module, nn.Conv2d):
                init.xavier_uniform_(m.module.weight, math.sqrt(2))
                init.zeros_(m.module.bias)
        for m in self.shortcut.modules():
            if isinstance(m, SpectralNorm) and isinstance(m.module, nn.Conv2d):
                init.xavier_uniform_(m.module.weight)
                init.zeros_(m.module.bias)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x)) / 2


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(SpectralNorm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
            shortcut.append(Scalar(4))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
            residual.append(Scalar(4))
        self.residual = nn.Sequential(*residual)
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, SpectralNorm) and isinstance(m.module, nn.Conv2d):
                init.xavier_uniform_(m.module.weight, math.sqrt(2))
                init.zeros_(m.module.bias)
        for m in self.shortcut.modules():
            if isinstance(m, SpectralNorm) and isinstance(m.module, nn.Conv2d):
                init.xavier_uniform_(m.module.weight)
                init.zeros_(m.module.bias)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x)) / 2


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = SpectralNorm(nn.Linear(128, 1, bias=False))
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.module.weight)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


if __name__ == '__main__':
    import copy
    import torch.nn as nn
    from tqdm import trange

    device = torch.device('cuda:0')

    def auto_spectral_norm(module, in_dim, iteration=100):
        v = normalize(torch.randn(size=in_dim)).to(device)
        with torch.no_grad():
            output_shape = module(v).shape
        v_dummy = torch.randn(size=in_dim, requires_grad=True).to(device)

        if isinstance(module, SpectralNorm):
            bias = getattr(module.module, 'bias')
        else:
            bias = getattr(module, 'bias')
        while len(bias.shape) < len(output_shape) - 1:
            bias = bias.unsqueeze(-1)

        for _ in trange(iteration):
            with torch.no_grad():
                if bias is not None:
                    u = normalize(module(v) - bias)
                else:
                    u = normalize(module(v))
            v = torch.autograd.grad(module(v_dummy), v_dummy, u)[0]
            with torch.no_grad():
                v = normalize(v)

        with torch.no_grad():
            if bias is not None:
                return torch.dot(u.reshape(-1), (module(v) - bias).reshape(-1))
            else:
                return torch.dot(u.reshape(-1), module(v).reshape(-1))

    in_dim = (1, 8, 32, 32)
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=True).to(device)
    sn_conv = SpectralNorm(copy.deepcopy(conv), n_iteration=100).to(device)
    # sn_conv = spectral_norm(
    #     copy.deepcopy(conv), n_power_iterations=100).to(device)

    print('Conv2d: %.4f' % auto_spectral_norm(conv, in_dim))
    print('SN Conv2d: %.4f' % auto_spectral_norm(sn_conv, in_dim))

    in_dim = (1, 5)
    linear = nn.Linear(5, 10).to(device)
    sn_linear = SpectralNorm(copy.deepcopy(linear), n_iteration=100).to(device)

    print('Linear: %.4f' % auto_spectral_norm(linear, in_dim))
    print('SN Linear: %.4f' % auto_spectral_norm(sn_linear, in_dim))
