import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3 * resolution * resolution),
            nn.Tanh(),
        )
        self.resolution = resolution

    def forward(self, z, *args, **kwargs):
        x = self.main(z)
        x = x.view(-1, 3, self.resolution, self.resolution)
        return x


class Discriminator(nn.Module):
    def __init__(self, resolution, n_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3 * resolution * resolution, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x, *args, **kwargs):
        y = self.main(x)
        return y
