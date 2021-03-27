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
