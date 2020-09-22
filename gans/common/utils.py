import os
import random
from contextlib import contextmanager

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import trange


device = torch.device('cuda:0')


def generate_imgs(net_G, z_dim=128, num_images=50000, batch_size=128):
    imgs = []
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            z = torch.randn(end - start, z_dim).to(device)
            imgs.append(net_G(z).cpu().numpy())
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs


def generate_conditional_imgs(net_G, n_classes=10, z_dim=128, num_images=50000,
                              batch_size=128):
    imgs = []
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            z = torch.randn(end - start, z_dim).to(device)
            y = torch.randint(n_classes, size=(end - start,)).to(device)
            imgs.append(net_G(z, y).cpu().numpy())
    imgs = np.concatenate(imgs, axis=0)
    imgs = (imgs + 1) / 2
    return imgs


def generate_and_save(net_G, output_dir, z_dim=128, num_images=50000,
                      batch_size=128):
    counter = 0
    os.makedirs(output_dir)
    with torch.no_grad():
        for start in trange(0, num_images, batch_size, dynamic_ncols=True):
            batch_size = min(batch_size, num_images - start)
            z = torch.randn(batch_size, z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(image, os.path.join(output_dir, '%d.png' % counter))
                counter += 1


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])
