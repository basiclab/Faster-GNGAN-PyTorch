from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradnorm import rescale_module


sn = partial(torch.nn.utils.spectral_norm, eps=1e-6)


class Attention(nn.Module):
    """
    SA-GAN: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, ch, use_spectral_norm):
        super().__init__()
        if use_spectral_norm:
            spectral_norm = sn
        else:
            spectral_norm = (lambda x: x)
        self.q = spectral_norm(nn.Conv2d(
            ch, ch // 8, kernel_size=1, padding=0, bias=False))
        self.k = spectral_norm(nn.Conv2d(
            ch, ch // 8, kernel_size=1, padding=0, bias=False))
        self.v = spectral_norm(nn.Conv2d(
            ch, ch // 2, kernel_size=1, padding=0, bias=False))
        self.o = spectral_norm(nn.Conv2d(
            ch // 2, ch, kernel_size=1, padding=0, bias=False))
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        B, C, H, W = x.size()
        q = self.q(x)
        k = F.max_pool2d(self.k(x), [2, 2])
        v = F.max_pool2d(self.v(x), [2, 2])
        # flatten
        q = q.view(B, C // 8, H * W)            # query
        k = k.view(B, C // 8, H * W // 4)       # key
        v = v.view(B, C // 2, H * W // 4)       # value
        # attention weights
        w = F.softmax(torch.bmm(q.transpose(1, 2), k), -1)
        # attend and project
        o = self.o(torch.bmm(v, w.transpose(1, 2)).view(B, C // 2, H, W))
        return self.gamma * o + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channel, cond_size, linear=True):
        super().__init__()
        if linear:
            self.gain = sn(nn.Linear(cond_size, in_channel, bias=False))
            self.bias = sn(nn.Linear(cond_size, in_channel, bias=False))
        else:
            self.gain = nn.Embedding(cond_size, in_channel)
            self.bias = nn.Embedding(cond_size, in_channel)
        self.batchnorm2d = nn.BatchNorm2d(in_channel, affine=False)

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1) + 1
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = self.batchnorm2d(x)
        return x * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cbn_in_dim, cbn_linear=True):
        """
            cbn_in_dim(int): output size of shared embedding
            cbn_linear(bool): use linear layer in conditional batchnorm to
                              get gain and bias of normalization. Otherwise,
                              use embedding.
        """
        super().__init__()

        # residual
        self.bn1 = ConditionalBatchNorm2d(in_channels, cbn_in_dim, cbn_linear)
        self.residual1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            sn(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)))
        self.bn2 = ConditionalBatchNorm2d(out_channels, cbn_in_dim, cbn_linear)
        self.residual2 = nn.Sequential(
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)))

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            sn(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)))

    def forward(self, x, y):
        h = self.residual1(self.bn1(x, y))
        h = self.residual2(self.bn2(h, y))
        return h + self.shortcut(x)


class Generator32(nn.Module):
    def __init__(self, z_dim=128, n_classes=10, ch=64):
        super().__init__()
        # channels_multipler = [4, 4, 4, 4]
        self.linear = sn(nn.Linear(z_dim, (ch * 4) * 4 * 4))
        self.blocks = nn.ModuleList([
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 8 x 8
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 16 x 16
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 32 x 32
        ])
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(ch * 4, 3, 3, padding=1)),      # 3 x 32 x 32
            nn.Tanh())
        res32_weights_init(self)

    def forward(self, z, y):
        h = self.linear(z).view(z.size(0), -1, 4, 4)
        for block in self.blocks:
            h = block(h, y)
        h = self.output_layer(h)
        return h


class Generator128(nn.Module):
    def __init__(self, z_dim=128, n_classes=1000, ch=96, shared_dim=128):
        super().__init__()
        channels_multipler = [16, 16, 8, 4, 2, 1]
        num_slots = len(channels_multipler)
        self.chunk_size = (z_dim // num_slots)
        z_dim = self.chunk_size * num_slots
        cbn_in_dim = (shared_dim + self.chunk_size)

        self.shared_embedding = nn.Embedding(n_classes, shared_dim)
        self.linear = sn(nn.Linear(z_dim // num_slots, (ch * 16) * 4 * 4))

        self.blocks = nn.ModuleList([
            GenBlock(ch * 16, ch * 16, cbn_in_dim),  # ch*16 x 4 x 4
            GenBlock(ch * 16, ch * 8, cbn_in_dim),   # ch*16 x 8 x 8
            GenBlock(ch * 8, ch * 4, cbn_in_dim),    # ch*8 x 16 x 16
            nn.ModuleList([                          # ch*4 x 32 x 32
                GenBlock(ch * 4, ch * 2, cbn_in_dim),
                Attention(ch * 2, True),             # ch*2 x 64 x 64
            ]),
            GenBlock(ch * 2, ch * 1, cbn_in_dim),    # ch*1 x 128 x 128
        ])

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(ch * 1),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(ch * 1, 3, 3, padding=1)),  # 3 x 128 x 128
            nn.Tanh())
        # res128_weights_init(self)

    def forward(self, z, y):
        y = self.shared_embedding(y)
        zs = torch.split(z, self.chunk_size, 1)
        ys = [torch.cat([y, item], 1) for item in zs[1:]]

        h = self.linear(zs[0]).view(z.size(0), -1, 4, 4)
        for i, block in enumerate(self.blocks):
            if isinstance(block, nn.ModuleList):
                for module in block:
                    h = module(h, ys[i])
            else:
                h = block(h, ys[i])
        h = self.output_layer(h)

        return h


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
            nn.Conv2d(in_channels, out_channels, 1, padding=0))
        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.AvgPool2d(2))


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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)


class ReScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_scale = 1

    def rescale_model(self, min_norm=1.0, max_norm=1.33):
        base_scale = 1
        for block in self.model:
            if isinstance(block, ReScaleBlock):
                # print(base_scale)
                base_scale = block.rescale_block(base_scale)
        linear_scale, _ = rescale_module(self.linear, base_scale)
        embedding_scale, _ = rescale_module(self.embedding, base_scale)
        self.embedding_scale *= linear_scale / embedding_scale
        return linear_scale

    def forward(self, x, y):
        h = self.model(x).sum(dim=[2, 3])
        h = (
            self.linear(h) +
            (self.embedding(y) * h * self.embedding_scale).sum(
                dim=1, keepdim=True)
        )
        return h


class Discriminator32(ReScaleModel):
    def __init__(self, n_classes=10, ch=64):
        super().__init__()
        self.model = nn.Sequential(                 # 3 x 32 x 32
            OptimizedDisblock(3, ch * 4),           # ch*4 x 16 x 16
            DisBlock(ch * 4, ch * 4, down=True),    # ch*4 x 8 x 8
            DisBlock(ch * 4, ch * 4),               # ch*4 x 8 x 8
            DisBlock(ch * 4, ch * 4),               # ch*4 x 8 x 8
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Linear(ch * 4, 1)
        self.embedding = nn.Embedding(n_classes, ch * 4)
        res32_weights_init(self)


class Discriminator128(ReScaleModel):
    def __init__(self, n_classes=1000, ch=96):
        super().__init__()
        # channels_multipler = [1, 2, 4, 8, 16, 16]
        self.model = nn.Sequential(                # 3 x 128 x 128
            OptimizedDisblock(3, ch * 1),          # ch*1 x 64 x 64
            Attention(ch, False),                  # ch*1 x 64 x 64
            DisBlock(ch * 1, ch * 2, down=True),   # ch*2 x 32 x 32
            DisBlock(ch * 2, ch * 4, down=True),   # ch*4 x 16 x 16
            DisBlock(ch * 4, ch * 8, down=True),   # ch*8 x 8 x 8
            DisBlock(ch * 8, ch * 16, down=True),  # ch*16 x 4 x 4
            DisBlock(ch * 16, ch * 16),            # ch*16 x 4 x 4
            nn.ReLU(inplace=True),                 # ch*16 x 4 x 4
        )

        self.linear = nn.Linear(ch * 16, 1)
        self.embedding = nn.Embedding(n_classes, ch * 16)
        res128_weights_init(self)


def res32_weights_init(m):
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def res128_weights_init(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)


if __name__ == '__main__':
    Models = [
        (32, 10, Discriminator32),
        (128, 1000, Discriminator128),
    ]
    for res, n_classes, Model in Models:
        print("=" * 80)
        print(Model.__name__)
        x = torch.randn(2, 3, res, res, requires_grad=True).cuda()
        y = torch.randint(n_classes, (2,)).cuda()
        net_D = Model(n_classes).cuda()
        f = net_D(x, y=y)
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
            f_scaled = net_D(x, y=y)
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
