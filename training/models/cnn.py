import torch
import torch.nn as nn


from training.models.base import RescalableSequentialModel, RescalableWrapper


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
