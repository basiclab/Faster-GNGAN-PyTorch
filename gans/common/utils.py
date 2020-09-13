import random
from contextlib import contextmanager

import torch
import numpy as np


def generate_imgs(net_G, device, z_dim=128, size=5000, batch_size=128):
    net_G.eval()
    imgs = []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim).to(device)
            imgs.append(net_G(z).cpu().numpy())
    net_G.train()
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs


def generate_conditional_imgs(net_G, device, n_classes=10, z_dim=128,
                              size=5000, batch_size=128):
    net_G.eval()
    imgs = []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim).to(device)
            y = torch.randint(n_classes, size=(end - start,)).to(device)
            imgs.append(net_G(z, y).cpu().numpy())
    net_G.train()
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs


@contextmanager
def module_require_grad(m: torch.nn.Module, require_grad):
    name_to_require_grad = {}
    for name, param in m.named_parameters():
        name_to_require_grad[name] = param.requires_grad
        param.requires_grad_(require_grad)
    yield m
    for name, param in m.named_parameters():
        require_grad = name_to_require_grad[name]
        param.requires_grad_(require_grad)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
