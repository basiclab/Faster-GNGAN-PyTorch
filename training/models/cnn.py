import torch
import torch.nn as nn


from training.models.base import (
    RescalableSequentialModel, RescalableWrapper, Reshape)


class Generator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim, num_layers):
        super().__init__()
        assert resolution in [32, 48], "The resolution %d is not supported in Generator." % resolution

        M = 32 // (2 ** min(max(2, num_layers), 3))
        num_blocks = [num_layers // 3 + (stage < num_layers % 3) for stage in range(3)]
        num_channels = [256, 128, 64]
        now_ch = 512
        blocks = [
            nn.Linear(z_dim, M * M * now_ch),
            Reshape(-1, now_ch, M, M),
        ]
        for ch, num_block in zip(num_channels, num_blocks):
            if num_block == 0:
                continue
            blocks.append(nn.BatchNorm2d(now_ch))
            blocks.append(nn.ReLU(True))
            blocks.append(nn.ConvTranspose2d(
                now_ch, ch, kernel_size=4, stride=2, padding=1))
            now_ch = ch
            for _ in range(num_block - 1):
                blocks.append(nn.BatchNorm2d(now_ch))
                blocks.append(nn.ReLU(True))
                blocks.append(nn.ConvTranspose2d(
                    now_ch, ch, kernel_size=3, stride=1, padding=1))
                now_ch = ch
        blocks.append(nn.BatchNorm2d(now_ch))
        blocks.append(nn.ReLU(True))
        blocks.append(nn.ConvTranspose2d(
            now_ch, 3, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.Tanh())
        self.main = nn.Sequential(*blocks)

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.main(z)
        return x


class Generator2(Generator):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__(resolution, n_classes, z_dim, num_layers=2)


class Generator4(Generator):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__(resolution, n_classes, z_dim, num_layers=4)


class Generator6(Generator):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__(resolution, n_classes, z_dim, num_layers=6)


class DiscriminatorK(RescalableSequentialModel):
    def __init__(self, resolution, n_classes, num_layers=3):
        super().__init__()
        config = {
            32: 4,
            48: 6,
        }
        assert resolution in config, "The resolution %d is not supported in Generator." % resolution
        M = config[resolution]

        num_blocks = [num_layers // 3 + (stage < num_layers % 3) for stage in range(3)]
        num_channels = [64, 128, 256]
        now_ch = 3
        blocks = []
        for ch, num_block in zip(num_channels, num_blocks):
            for _ in range(num_block - 1):
                blocks.append(RescalableWrapper(nn.Conv2d(
                    now_ch, ch, kernel_size=3, stride=1, padding=1)))
                blocks.append(nn.LeakyReLU(0.1, inplace=True))
                now_ch = ch
            blocks.append(RescalableWrapper(nn.Conv2d(
                now_ch, ch * 2, kernel_size=4, stride=2, padding=1)))
            blocks.append(nn.LeakyReLU(0.1, inplace=True))
            now_ch = ch * 2
        blocks.append(nn.Flatten(start_dim=1))
        blocks.append(RescalableWrapper(nn.Linear(M * M * 512, 1)))
        self.main = nn.Sequential(*blocks)

        for m in self.modules():
            if isinstance(m, RescalableWrapper):
                torch.nn.init.normal_(m.module.weight, std=0.02)
                torch.nn.init.zeros_(m.module.bias)
                m.init_module()

    def forward(self, x, *args, **kwargs):
        y = self.main(x)
        return y


class Discriminator3(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_layers=3)


class Discriminator5(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_layers=5)


class Discriminator7(DiscriminatorK):
    def __init__(self, resolution, n_classes):
        super().__init__(resolution, n_classes, num_layers=7)


if __name__ == '__main__':
    G = Generator2(32, None, 128)
    print(G(torch.rand(64, 128)).shape)
    print(G)
    G = Generator4(32, None, 128)
    print(G(torch.rand(64, 128)).shape)
    print(G)
    G = Generator6(32, None, 128)
    print(G(torch.rand(64, 128)).shape)
    print(G)
    print(Discriminator3(32, None))
    print(Discriminator5(32, None))
    print(Discriminator7(32, None))
