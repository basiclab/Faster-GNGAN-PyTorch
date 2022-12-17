import torch
import torch.nn as nn
import torch.nn.init as init

from tqdm import trange


def normalize(x):
    norm = torch.norm(torch.flatten(x))
    return x / (norm + 1e-12)


class SpectralNorm(nn.Module):
    def __init__(self, module, n_iteration=10):
        assert isinstance(module, (nn.Conv2d, nn.Linear)), \
            'module must be nn.Conv2d or nn.Linear'
        super().__init__()
        self.module = module
        self.n_iteration = n_iteration
        self.register_buffer('u', None)
        self.register_buffer('v', None)

    def forward(self, x):
        # weight = self.module.weight.view(self.module.weight.size(0), -1)
        if self.u is None:
            self.v = normalize(torch.randn_like(x[0])).unsqueeze(0)
            with torch.no_grad():
                self.u = normalize(self._forward(self.v))
            initial_forward = True
        else:
            initial_forward = False
        if self.training:
            with torch.no_grad():
                u = self.u
                if initial_forward:
                    # increase the number of iteration for the first forward
                    n_iteration = self.n_iteration * 10
                else:
                    n_iteration = self.n_iteration
                for _ in range(n_iteration):
                    v = normalize(self._backward(u))
                    u = normalize(self._forward(v))
                if self.n_iteration > 0:
                    self.u = u.clone(memory_format=torch.contiguous_format)
                    self.v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.sum(self.v * self._backward(self.u))
        return self._forward(
            x, self.module.weight / sigma, self.module.bias)

    def _forward(self, x, weight=None, bias=None):
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

    def _backward(self, x, weight=None, bias=None):
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
    def __init__(self, resolution, n_classes, num_per_block=1):
        super().__init__()
        config = {
            32: 4,
            48: 6,
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        M = config[resolution]

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
        blocks.append(nn.Flatten(start_dim=1))
        blocks.append(SpectralNorm(nn.Linear(M * M * 512, 1)))
        self.main = nn.Sequential(*blocks)

        for m in self.modules():
            if isinstance(m, SpectralNorm):
                if isinstance(m.module, (nn.Conv2d, nn.Linear)):
                    init.normal_(m.module.weight, std=0.02)
                    init.zeros_(m.module.bias)

    def forward(self, x, *args, **kwargs):
        y = self.main(x)
        return y


class Discriminator3(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_per_block=1)


class Discriminator6(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_per_block=2)


class Discriminator9(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_per_block=3)


def auto_spectral_norm(module, in_dim, iteration=100, device=torch.device('cuda:0')):
    v = normalize(torch.randn(size=in_dim)).to(device)
    with torch.no_grad():
        output_shape = module(v).shape
    v_dummy = torch.randn(size=in_dim, requires_grad=True).to(device)

    # get bias
    if isinstance(module, SpectralNorm):
        bias = getattr(module.module, 'bias')
    else:
        bias = getattr(module, 'bias')
    while len(bias.shape) < len(output_shape) - 1:
        bias = bias.unsqueeze(-1)

    # power iteration
    for _ in trange(iteration, leave=False, ncols=0, desc="power iteration"):
        with torch.no_grad():
            if bias is not None:
                u = normalize(module(v) - bias)
            else:
                u = normalize(module(v))
        v = torch.autograd.grad(module(v_dummy), v_dummy, u)[0]
        with torch.no_grad():
            v = normalize(v)

    # calculate spectral norm
    with torch.no_grad():
        if bias is not None:
            return torch.dot(u.reshape(-1), (module(v) - bias).reshape(-1))
        else:
            return torch.dot(u.reshape(-1), module(v).reshape(-1))


if __name__ == '__main__':
    """
    The following program is used to test the class `SpectralNorm`.
    """
    import copy
    from torch.nn.utils import spectral_norm

    device = torch.device('cuda:0')

    in_dim = (1, 8, 32, 32)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True).to(device)
    sn_conv = SpectralNorm(copy.deepcopy(conv), n_iteration=100).to(device)
    sn_torch_conv = spectral_norm(copy.deepcopy(conv), n_power_iterations=100).to(device)

    print(f'{"Conv2d": >15s}: {auto_spectral_norm(conv, in_dim):.4f}')
    print(f'{"SN Conv2d": >15s}: {auto_spectral_norm(sn_conv, in_dim):.4f}')
    print(f'{"SN_torch Conv2d": >15s}: {auto_spectral_norm(sn_torch_conv, in_dim):.4f}')

    in_dim = (1, 5)
    linear = nn.Linear(5, 10).to(device)
    sn_linear = SpectralNorm(copy.deepcopy(linear), n_iteration=100).to(device)
    sn_torch_linear = spectral_norm(copy.deepcopy(linear), n_power_iterations=100).to(device)

    print(f'{"Linear": >15s}: {auto_spectral_norm(linear, in_dim):.4f}')
    print(f'{"SN Linear": >15s}: {auto_spectral_norm(sn_linear, in_dim):.4f}')
    print(f'{"SN_torch Linear": >15s}: {auto_spectral_norm(sn_torch_linear, in_dim):.4f}')

    # sanity check
    Discriminators = (Discriminator3, Discriminator6, Discriminator9)
    for Discriminators in Discriminators:
        for resolution in [32, 48]:
            D = Discriminators(resolution, None).to(device)
            for _ in range(10):
                x1 = torch.randn(
                    size=(128, 3, resolution, resolution), requires_grad=True
                ).to(device)
                x2 = x1 + 0.02 * torch.randn_like(x1)
                y1 = D(x1)
                y2 = D(x2)

                x1_grad = torch.autograd.grad(y1.sum(), x1)[0]
