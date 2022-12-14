import torch
import torch.nn as nn


from training.models.base import RescalableSequentialModel, RescalableWrapper


class DiscriminatorK(RescalableSequentialModel):
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
