import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channel, n_classes):
        super().__init__()
        self.gain = nn.Embedding(n_classes, in_channel)
        self.bias = nn.Embedding(n_classes, in_channel)
        self.register_buffer('stored_mean', torch.zeros(in_channel))
        self.register_buffer('stored_var',  torch.ones(in_channel))
        self.initialize()

    def initialize(self):
        init.ones_(self.gain.weight)
        init.zeros_(self.bias.weight)

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = F.batch_norm(
            x, self.stored_mean, self.stored_var, None, None, self.training)
        return x * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0))
        # residual
        self.bn1 = ConditionalBatchNorm2d(in_channels, n_classes)
        self.cnn1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.bn2 = ConditionalBatchNorm2d(out_channels, n_classes)
        self.cnn2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in list(self.cnn1.modules()) + list(self.cnn2.modules()):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x, y):
        h = self.cnn1(self.bn1(x, y))
        h = self.cnn2(self.bn2(h, y))
        return h + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, ch, n_classes, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * ch * 4)
        self.blocks = nn.ModuleList([
            GenBlock(ch * 4, ch * 4, n_classes),
            GenBlock(ch * 4, ch * 4, n_classes),
            GenBlock(ch * 4, ch * 4, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(),
            nn.Conv2d(ch * 4, 3, 3, stride=1, padding=1),
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

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(z.shape[0], -1, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class ResGenerator128(nn.Module):
    def __init__(self, ch, n_classes, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * ch * 16)

        self.blocks = nn.ModuleList([
            GenBlock(ch * 16, ch * 16, n_classes),
            GenBlock(ch * 16, ch * 8, n_classes),
            GenBlock(ch * 8, ch * 4, n_classes),
            GenBlock(ch * 4, ch * 2, n_classes),
            GenBlock(ch * 2, ch, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, 3, 3, stride=1, padding=1),
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

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(z.shape[0], -1, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class OptimizedDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # shortcut
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        # residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
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
        # initialize weight
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDiscriminator32(nn.Module):
    def __init__(self, ch, n_classes):
        super().__init__()
        self.main = nn.Sequential(
            OptimizedDisblock(3, ch * 2),
            DisBlock(ch * 2, ch * 2, down=True),
            DisBlock(ch * 2, ch * 2),
            DisBlock(ch * 2, ch * 2),
            nn.ReLU(inplace=True))
        self.linear = nn.Linear(ch * 2, 1, bias=False)
        self.embed = nn.Embedding(n_classes, ch * 2)
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.kaiming_normal_(self.embed.weight)

    def forward(self, x, y):
        x = self.main(x).sum(dim=[2, 3])
        x = self.linear(x) + (self.embed(y) * x).sum(dim=1, keepdim=True)
        return x


class ResConcatDiscriminator128(nn.Module):
    def __init__(self, ch, n_classes):
        super().__init__()
        self.main1 = nn.Sequential(
            OptimizedDisblock(3, ch),
            DisBlock(ch, ch * 2, down=True),
            DisBlock(ch * 2, ch * 4, down=True))
        self.embed = nn.Embedding(n_classes, 128)
        self.main2 = nn.Sequential(
            DisBlock(ch * 4 + 128, ch * 8, down=True),
            DisBlock(ch * 8, ch * 16, down=True),
            DisBlock(ch * 16, ch * 16),
            nn.ReLU(inplace=True))
        self.linear = nn.Linear(ch * 16, 1)
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        init.kaiming_normal_(self.embed.weight)

    def forward(self, x, y):
        x = self.main1(x)
        e = self.embed(y).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, e.expand(-1, -1, x.shape[2], x.shape[2])], dim=1)
        x = self.main2(x).sum([2, 3])
        x = self.linear(x)
        return x


class ResPorjectDiscriminator128(nn.Module):
    def __init__(self, ch, n_classes):
        super().__init__()
        self.main = nn.Sequential(
            OptimizedDisblock(3, ch),
            DisBlock(ch, ch * 2, down=True),
            DisBlock(ch * 2, ch * 4, down=True),
            DisBlock(ch * 4, ch * 8, down=True),
            DisBlock(ch * 8, ch * 16, down=True),
            DisBlock(ch * 16, ch * 16),
            nn.ReLU(inplace=True))
        self.embed = nn.Embedding(n_classes, ch * 16)
        self.linear = nn.Linear(ch * 16, 1)
        self.initialize()

    def initialize(self):
        init.kaiming_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)
        init.kaiming_normal_(self.embed.weight)

    def forward(self, x, y):
        x = self.main(x).sum(dim=[2, 3])
        x = self.linear(x) + (self.embed(y) * x).sum(dim=1, keepdim=True)
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
