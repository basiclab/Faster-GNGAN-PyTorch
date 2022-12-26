import os
import glob

import torch
from torchvision.utils import save_image
from torchvision.io.image import read_image


if __name__ == '__main__':
    save_dir = './vis/figures'
    sample_dirs = {
        # 'celebahq256.png': 'logs/GN_celebahq256_resnet_rescale1_0/generate',
        'church256.png': 'logs/GN_church256_resnet_rescale0_xavier_0/generate',
    }
    grid = 3

    for save_name, sample_dir in sample_dirs.items():
        save_path = os.path.join(save_dir, save_name)
        files = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
        indexs = torch.randperm(len(files))[:grid * grid]
        files = [files[index] for index in indexs]
        images = torch.stack([read_image(f) for f in files]) / 255.
        save_image(images, save_path, nrow=grid)
