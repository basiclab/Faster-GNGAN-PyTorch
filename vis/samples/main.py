import os
import glob

import torch
from torchvision.utils import save_image
from torchvision.io.image import read_image


if __name__ == '__main__':
    save_dir = './vis/figures'
    sample_dirs = {
        'celebahq256': (
            'logs/GN_celebahq256_resnet_rescale1_0/generate',
            [11144, 40665, 8724, 29462, 5320, 7059, 44644]
            # []
        ),
        'church256': (
            'logs/GN_church256_resnet_rescale0_xavier_0/generate',
            [17692, 37405, 46852, 24441, 25746, 9882, 16996]
            # []
        ),
    }

    row_length = 7
    images = []
    names = []
    for name, (sample_dir, init_indexs) in sample_dirs.items():
        files = sorted(glob.glob(os.path.join(sample_dir, '*.png')))
        if len(init_indexs) < row_length:
            indexs = torch.randperm(len(files))[:row_length - len(init_indexs)]
            indexs = torch.cat([indexs, torch.tensor(init_indexs).long()], dim=0)
        else:
            indexs = torch.tensor(init_indexs).long()
        print(indexs)
        files = [files[index] for index in indexs]
        images.extend([read_image(f) for f in files])
        names.append(name)
    images = torch.stack(images) / 255.
    save_path = os.path.join(save_dir, ".".join(names) + ".png")
    save_image(images, save_path, nrow=row_length)
