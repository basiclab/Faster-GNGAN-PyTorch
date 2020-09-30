import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Attention(nn.Module):
    """
    SA-GAN: https://arxiv.org/abs/1805.08318
    """
    def __init__(self, ch, sn=spectral_norm):
        super().__init__()
        self.f = sn(nn.Conv2d(
            ch, ch // 8, kernel_size=1, padding=0, bias=False))
        self.g = sn(nn.Conv2d(
            ch, ch // 8, kernel_size=1, padding=0, bias=False))
        self.h = sn(nn.Conv2d(
            ch, ch // 2, kernel_size=1, padding=0, bias=False))
        self.v = sn(nn.Conv2d(
            ch // 2, ch, kernel_size=1, padding=0, bias=False))
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        B, C, H, W = x.size()
        f = self.f(x)
        g = F.max_pool2d(self.g(x), [2, 2])
        h = F.max_pool2d(self.h(x), [2, 2])
        # flatten
        f = f.view(B, C // 8, H * W)
        g = g.view(B, C // 8, H * W // 4)
        h = h.view(B, C // 2, H * W // 4)
        # attention weights
        w = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        # attend and project
        o = self.v(torch.bmm(h, w.transpose(1, 2)).view(B, C // 2, H, W))
        return self.gamma * o + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channel, cond_size, use_linear=True,
                 sn=spectral_norm):
        super().__init__()
        if use_linear:
            self.gain = sn(nn.Linear(cond_size, in_channel, bias=False))
            self.bias = sn(nn.Linear(cond_size, in_channel, bias=False))
        else:
            self.gain = nn.Embedding(cond_size, in_channel)
            self.bias = nn.Embedding(cond_size, in_channel)
        self.register_buffer('stored_mean', torch.zeros(in_channel))
        self.register_buffer('stored_var',  torch.ones(in_channel))
        # self.batchnorm = nn.BatchNorm2d(in_channel, affine=False)

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1) + 1
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = F.batch_norm(
            x, self.stored_mean, self.stored_var, None, None, self.training)
        # x = self.batchnorm(x)
        return x * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shared_input_size,
                 use_linear=True):
        """
            shared_input_size(int): output size of shared embedding
            use_linear(bool): use linear layer in conditional batchnorm to
                              project shared embedding to gain and bias.
                              Otherwise, use embedding in conditional batchnorm
        """
        super().__init__()

        self.shorcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, padding=0)))

        # residual
        self.bn1 = ConditionalBatchNorm2d(
            in_channels, shared_input_size, use_linear)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.bn2 = ConditionalBatchNorm2d(
            out_channels, shared_input_size, use_linear)
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x, y):
        # residual
        h = self.activation(self.bn1(x, y))
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        # shorcut
        x = self.shorcut(x)
        return h + x


class Generator32(nn.Module):
    def __init__(self, ch=64, n_classes=10, z_dim=128):
        super().__init__()
        # channels_multipler = [4, 4, 4, 4]
        self.linear = spectral_norm(nn.Linear(z_dim, (ch * 4) * 4 * 4))
        self.blocks = nn.ModuleList([
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 8 x 8
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 16 x 16
            GenBlock(ch * 4, ch * 4, n_classes, False),  # 4ch x 32 x 32
        ])
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
            spectral_norm(
                nn.Conv2d(ch * 4, 3, 3, padding=1)),     # 3 x 32 x 32
            nn.Tanh())
        self.shared = nn.Identity()
        # if not kwargs['skip_init']:
        weights_init(self)

    def forward(self, z, y):
        h = self.linear(z).view(z.size(0), -1, 4, 4)
        for block in self.blocks:
            h = block(h, y)
        h = self.output_layer(h)
        return h


class Generator128(nn.Module):
    def __init__(self, ch=96, n_classes=1000, z_dim=128, shared_dim=128):
        super().__init__()
        channels_multipler = [16, 16, 8, 4, 2, 1]
        num_slots = len(channels_multipler)
        self.chunk_size = (z_dim // num_slots)
        z_dim = self.chunk_size * num_slots
        shared_input_size = (shared_dim + self.chunk_size)

        self.shared_embedding = nn.Embedding(n_classes, shared_dim)
        self.linear = spectral_norm(
            nn.Linear(z_dim // num_slots, (ch * 16) * 4 * 4))

        self.blocks = nn.ModuleList([
            GenBlock(ch * 16, ch * 16, shared_input_size),  # ch*16 x 4 x 4
            GenBlock(ch * 16, ch * 8, shared_input_size),   # ch*16 x 8 x 8
            GenBlock(ch * 8, ch * 4, shared_input_size),    # ch*8 x 16 x 16
            nn.ModuleList([                                 # ch*4 x 32 x 32
                GenBlock(ch * 4, ch * 2, shared_input_size),
                Attention(ch * 2),                          # ch*2 x 64 x 64
            ]),
            GenBlock(ch * 2, ch * 1, shared_input_size),    # ch*1 x 128 x 128
        ])

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(ch * 1),
            nn.ReLU(inplace=True),
            spectral_norm(
                nn.Conv2d(ch * 1, 3, 3, padding=1)),        # 3 x 128 x 128
            nn.Tanh())
        weights_init(self)

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


class OptimizedDisblock(nn.Module):
    def __init__(self, in_channels, out_channels, sn=spectral_norm):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            sn(nn.Conv2d(in_channels, out_channels, 1, padding=0)))
        # residual
        self.residual = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False,
                 sn=spectral_norm):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(sn(nn.Conv2d(in_channels, out_channels, 1, 1, 0)))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            sn(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.ReLU(inplace=True),
            sn(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator32(nn.Module):
    def __init__(self, ch=64, n_classes=10, sn=spectral_norm):
        super().__init__()
        self.fp16 = False
        # channels_multipler = [2, 2, 2, 2]
        self.blocks = nn.Sequential(
            OptimizedDisblock(3, ch * 4, sn=sn),           # 3 x 32 x 32
            DisBlock(ch * 4, ch * 4, down=True, sn=sn),    # ch*4 x 16 x 16
            DisBlock(ch * 4, ch * 4, sn=sn),               # ch*4 x 8 x 8
            DisBlock(ch * 4, ch * 4, sn=sn),               # ch*4 x 8 x 8
            nn.ReLU(inplace=True),
        )

        self.linear = sn(nn.Linear(ch * 4, 1))
        self.embedding = sn(nn.Embedding(n_classes, ch * 4))
        # if not kwargs['skip_init']:
        weights_init(self)

    def forward(self, x, y):
        h = self.blocks(x).sum(dim=[2, 3])
        h = self.linear(h) + (self.embedding(y) * h).sum(dim=1, keepdim=True)
        return h


class Discriminator128(nn.Module):
    def __init__(self, ch=96, n_classes=1000, sn=spectral_norm):
        super().__init__()
        # channels_multipler = [1, 2, 4, 8, 16, 16]
        self.blocks = nn.Sequential(
            OptimizedDisblock(3, ch * 1, sn=sn),          # 3 x 128 x 128
            Attention(ch, sn),                            # ch*1 x 64 x 64
            DisBlock(ch * 1, ch * 2, down=True, sn=sn),   # ch*1 x 32 x 32
            DisBlock(ch * 2, ch * 4, down=True, sn=sn),   # ch*2 x 16 x 16
            DisBlock(ch * 4, ch * 8, down=True, sn=sn),   # ch*4 x 8 x 8
            DisBlock(ch * 8, ch * 16, down=True, sn=sn),  # ch*8 x 4 x 4
            DisBlock(ch * 16, ch * 16, sn=sn),            # ch*16 x 4 x 4
            nn.ReLU(inplace=True),                        # ch*16 x 4 x 4
        )

        self.linear = sn(nn.Linear(ch * 16, 1))
        self.embedding = sn(nn.Embedding(n_classes, ch * 16))
        weights_init(self)

    def forward(self, x, y):
        h = self.blocks(x).sum(dim=[2, 3])
        h = self.linear(h) + (self.embedding(y) * h).sum(dim=1, keepdim=True)
        return h


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
                x_fake = self.net_G(z, self.net_G.shared(y_fake)).detach()
            x = torch.cat([x_real, x_fake], dim=0)
            y = torch.cat([y_real, y_fake], dim=0)
            pred = self.net_D(x, y=y)
            net_D_real, net_D_fake = torch.split(
                pred, [x_real.shape[0], x_fake.shape[0]])
            return net_D_real, net_D_fake
        else:
            x_fake = self.net_G(z, self.net_G.shared(y_fake))
            net_D_fake = self.net_D(x_fake, y=y_fake)
            return net_D_fake


def weights_init(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, 0, 0.02)
            # torch.nn.init.kaiming_normal_(module.weight.data)
            # if hasattr(module, 'bias') and module.bias is not None:
            #     torch.nn.init.zeros_(module.bias.data)
