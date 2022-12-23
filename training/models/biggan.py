from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Reshape, RescalableWrapper, RescalableResBlock, RescalableSequentialModel


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

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x)
        k = F.max_pool2d(self.k(x), [2, 2])
        v = F.max_pool2d(self.v(x), [2, 2])
        # flatten
        q = q.view(B, C // 8, H * W).contiguous()            # query
        k = k.view(B, C // 8, H * W // 4).contiguous()       # key
        v = v.view(B, C // 2, H * W // 4).contiguous()       # value
        # attention weights
        w = F.softmax(torch.bmm(q.transpose(1, 2), k), -1)
        # attend and project
        o = self.o(torch.bmm(v, w.transpose(1, 2)).view(B, C // 2, H, W))
        return self.gamma * o + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, channels, in_dim, linear=True):
        super().__init__()
        if linear:
            self.gain = sn(nn.Linear(in_dim, channels, bias=False))
            self.bias = sn(nn.Linear(in_dim, channels, bias=False))
        else:
            self.gain = nn.Embedding(in_dim, channels)
            self.bias = nn.Embedding(in_dim, channels)
        self.batchnorm2d = nn.BatchNorm2d(channels, affine=False)

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1) + 1
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = self.batchnorm2d(x)
        return x * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cbn_in_dim, cbn_linear=True):
        super().__init__()

        # main
        self.bn1 = ConditionalBatchNorm2d(in_channels, cbn_in_dim, cbn_linear)
        self.main1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.bn2 = ConditionalBatchNorm2d(out_channels, cbn_in_dim, cbn_linear)
        self.main2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0))

    def forward(self, x, y):
        h = self.main1(self.bn1(x, y))
        h = self.main2(self.bn2(h, y))
        return h + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim, shared_dim=128):
        super().__init__()
        config = {
            # channels, attn_indices, use_shared_embedding
            32: ([256, 256, 256, 256], [], False),
            128: ([1024, 1024, 512, 256, 128, 64], [4], True),  # ch = 64
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        channels, attn_indices, use_shared_embedding = config[resolution]

        # shared embedding for condition batchnorm (cbn)
        self.use_shared_embedding = use_shared_embedding
        if use_shared_embedding:
            self.z_chunk_num = len(channels)
            self.z_chunk_size = z_dim // self.z_chunk_num
            self.cbn_in_dim = self.z_chunk_size + shared_dim
            self.shared_embedding = nn.Embedding(n_classes, shared_dim)
            blocks = [sn(nn.Linear(self.z_chunk_size, channels[0] * 4 * 4))]
        else:
            # CIFAR10
            self.z_chunk_num = len(channels)    # for code simplicity
            self.cbn_in_dim = n_classes
            blocks = [sn(nn.Linear(z_dim, channels[0] * 4 * 4))]
        blocks.append(Reshape(-1, channels[0], 4, 4))

        for i in range(1, len(channels)):
            blocks.append(GenBlock(
                channels[i - 1],
                channels[i],
                self.cbn_in_dim,
                cbn_linear=use_shared_embedding))
            if i in attn_indices:
                blocks.append(Attention(channels[i], use_spectral_norm=True))
        blocks.extend([
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(channels[-1], 3, 3, padding=1)),
            nn.Tanh()
        ])

        self.blocks = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                torch.nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, z, y):
        if self.use_shared_embedding:
            y = self.shared_embedding(y)
            z = z.split(self.z_chunk_size, dim=1)[:self.z_chunk_num]
            z0 = z[0]
            y = [torch.cat([y, z_chunk], dim=1) for z_chunk in z[1:]]
        else:
            z0 = z
            y = [y for _ in range(self.z_chunk_num - 1)]

        x = z0
        y = iter(y)
        for m in self.blocks:
            if isinstance(m, GenBlock):
                x = m(x, next(y))
            else:
                x = m(x)

        return x


class OptimizedDisBlock(RescalableResBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 1, padding=0)))
        # main
        self.main = nn.Sequential(
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            nn.ReLU(inplace=True),
            RescalableWrapper(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
            nn.AvgPool2d(2))


class DisBlock(RescalableResBlock):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(RescalableWrapper(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)
        # main
        main = [
            nn.ReLU(),
            RescalableWrapper(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(inplace=True),
            RescalableWrapper(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            main.append(nn.AvgPool2d(2))
        self.main = nn.Sequential(*main)


class RescalableDiscriminator(RescalableSequentialModel):
    def __init__(self):
        super().__init__()
        self.emb_scale_gain = 1

    def rescale_model(self, base_scale=1., alpha=1.):
        base_scale = self.expand(self.main, base_scale, alpha)
        lin_scale = self.linear.rescale(base_scale, alpha)
        emb_scale = self.embedding.rescale(base_scale, alpha)
        self.emb_scale_gain = lin_scale / emb_scale
        return lin_scale

    def forward(self, x, y):
        e = self.embedding(y)
        h = self.main(x).sum(dim=[2, 3])
        y_lin = self.linear(h)
        y_cls = (e * h * self.emb_scale_gain).sum(dim=1, keepdim=True)
        return y_lin + y_cls


class Discriminator(RescalableDiscriminator):
    def __init__(self, resolution, n_classes):
        super().__init__()
        config = {
            # channels, attn_indices, down_layers
            32: ([256, 256, 256, 256], [], 2),
            128: ([64, 128, 256, 512, 1024, 1024], [0], 5),
        }
        assert resolution in config, "The resolution %d is not supported in Discriminator." % resolution
        channels, attn_indices, down_layers = config[resolution]

        for i in range(len(channels)):
            if i == 0:
                blocks = [OptimizedDisBlock(3, channels[i])]
            else:
                blocks.append(
                    DisBlock(channels[i - 1], channels[i], down=i < down_layers))
            if i in attn_indices:
                blocks.append(Attention(channels[i], use_spectral_norm=False))
        blocks.append(nn.ReLU(inplace=True)),

        self.main = nn.Sequential(*blocks)
        self.linear = RescalableWrapper(nn.Linear(channels[-1], 1))
        self.embedding = RescalableWrapper(nn.Embedding(n_classes, channels[-1]))

        for m in self.modules():
            if isinstance(m, RescalableWrapper):
                torch.nn.init.xavier_uniform_(m.module.weight)
                if hasattr(m.module, 'bias') and m.module.bias is not None:
                    torch.nn.init.zeros_(m.module.bias)
                m.init_module()
