import argparse
import glob
import os

import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate FID and inception score")
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--ncol', type=int, default=3)
    parser.add_argument('--nrow', type=int, default=3)
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    random_files = np.random.choice(
        files, args.ncol * args.nrow, replace=False)
    images = []
    for path in random_files:
        images.append(to_tensor(Image.open(path)))
    os.makedirs('figures', exist_ok=True)
    save_image(images, './figures/grid.png', nrow=args.nrow)
