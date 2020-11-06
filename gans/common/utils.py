import os
import random
from contextlib import contextmanager

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import trange, tqdm


device = torch.device('cuda:0')


def generate_images(net_G, z_dim, n_classes=None, num_images=10000,
                    batch_size=64, verbose=False):
    all_images = None
    with torch.no_grad():
        for start in trange(0, num_images, batch_size, disable=(not verbose)):
            batch_size = min(batch_size, num_images - start)
            z = torch.randn(batch_size, z_dim).to(device)
            # condition or unconditional
            if n_classes is not None:
                y = torch.randint(n_classes, size=(batch_size,)).to(device)
                images = net_G(z, y).cpu().numpy()
            else:
                images = net_G(z).cpu().numpy()
            if all_images is None:
                _, C, H, W = images.shape
                all_images = np.zeros((num_images, C, H, W))
            # yield or append to list
            all_images[start: start + len(images)] = images
    all_images = (all_images + 1) / 2
    return all_images


def images_generator(net_G, z_dim, n_classes=None, num_images=50000,
                     batch_size=64):
    """Python generator for generating large number of images"""
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            batch_size = min(batch_size, num_images - start)
            z = torch.randn(batch_size, z_dim).to(device)
            # condition or unconditional
            if n_classes is not None:
                y = torch.randint(n_classes, size=(batch_size,)).to(device)
                images = net_G(z, y).cpu().numpy()
            else:
                images = net_G(z).cpu().numpy()
            for image in images:
                yield (image + 1) / 2


def save_images(images, output_dir, verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(tqdm(images, disable=(not verbose))):
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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
