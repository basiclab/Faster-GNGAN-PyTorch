import os
import random
from contextlib import contextmanager

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import trange


device = torch.device('cuda:0')


def generate_images(net_G, z_dim=128, n_classes=None, num_images=10000,
                    batch_size=64, verbose=False):
    images_list = []
    with torch.no_grad():
        for start in trange(0, num_images, batch_size, disable=(not verbose)):
            batch_size = min(batch_size, num_images - start)
            z = torch.randn(batch_size, z_dim).to(device)
            if n_classes is not None:
                y = torch.randint(n_classes, size=(batch_size,)).to(device)
                images = net_G(z, y).cpu()
            else:
                images = net_G(z).cpu()
            images_list.append(images.numpy())
    images = np.concatenate(images_list, axis=0)
    images = (images + 1) / 2
    return images


def save_images(images, output_dir):
    for i, image in enumerate(images):
        save_image(torch.tensor(image), os.path.join(output_dir, '%d.png' % i))


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
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])
