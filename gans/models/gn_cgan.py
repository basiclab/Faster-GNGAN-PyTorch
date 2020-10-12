import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_channel, n_classes):
        super().__init__()
        self.gain = nn.Embedding(n_classes, in_channel)
        self.bias = nn.Embedding(n_classes, in_channel)
        self.register_buffer('stored_mean', torch.zeros(in_channel))
        self.register_buffer('stored_var',  torch.ones(in_channel))

    def forward(self, x, y):
        gain = self.gain(y).view(y.size(0), -1, 1, 1) + 1
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        x = F.batch_norm(
            x, self.stored_mean, self.stored_var, None, None, self.training)
        return x * gain + bias


class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()

        # residual
        self.bn1 = ConditionalBatchNorm2d(in_channels, n_classes)
        self.residual1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        self.bn2 = ConditionalBatchNorm2d(out_channels, n_classes)
        self.residual2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0))

    def forward(self, x, y):
        h = self.residual1(self.bn1(x, y))
        h = self.residual2(self.bn2(h, y))
        return h + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, n_classes, z_dim, ch=64):
        super().__init__()
        self.ch = ch
        self.linear = nn.Linear(z_dim, 4 * 4 * ch * 4)

        self.blocks = nn.ModuleList([
            ResGenBlock(ch * 4, ch * 4, n_classes),
            ResGenBlock(ch * 4, ch * 4, n_classes),
            ResGenBlock(ch * 4, ch * 4, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(),
            nn.Conv2d(ch * 4, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(-1, self.ch * 4, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class ResGenerator128(nn.Module):
    def __init__(self, n_classes, z_dim, ch=64):
        super().__init__()
        self.ch = ch
        self.linear = nn.Linear(z_dim, 4 * 4 * ch * 16)

        self.blocks = nn.ModuleList([
            ResGenBlock(ch * 16, ch * 16, n_classes),
            ResGenBlock(ch * 16, ch * 8, n_classes),
            ResGenBlock(ch * 8, ch * 4, n_classes),
            ResGenBlock(ch * 4, ch * 2, n_classes),
            ResGenBlock(ch * 2, ch, n_classes),
        ])
        self.output = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        weights_init(self)

    def forward(self, z, y):
        inputs = self.linear(z)
        inputs = inputs.view(-1, self.ch * 16, 4, 4)
        for module in self.blocks:
            inputs = module(inputs, y)
        return self.output(inputs)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self, n_classes, ch=64):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, ch * 2),
            ResDisBlock(ch * 2, ch * 2, down=True),
            ResDisBlock(ch * 2, ch * 2),
            ResDisBlock(ch * 2, ch * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(ch * 2, 1)
        self.embedding = nn.Embedding(n_classes, ch * 2)
        weights_init(self)

    def forward(self, x, y):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x) + (self.embedding(y) * x).sum(dim=1, keepdim=True)
        return x


class ResDiscriminator128(nn.Module):
    def __init__(self, n_classes, ch=64):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, ch),
            ResDisBlock(ch, ch * 2, down=True),
            ResDisBlock(ch * 2, ch * 4, down=True),
            ResDisBlock(ch * 4, ch * 8, down=True),
            ResDisBlock(ch * 8, ch * 16, down=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(ch * 16, 1)
        self.embedding = nn.Embedding(n_classes, ch * 16)
        weights_init(self)

    def forward(self, x, y):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x) + (self.embedding(y) * x).sum(dim=1, keepdim=True)
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


def weights_init(m):
    modules = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)
    for module in m.modules():
        if isinstance(module, modules):
            torch.nn.init.kaiming_normal_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)


generators = {
    'res32': ResGenerator32,
    'res128': ResGenerator128,
}

discriminators = {
    'res32': ResDiscriminator32,
    'res128': ResDiscriminator128,
}
