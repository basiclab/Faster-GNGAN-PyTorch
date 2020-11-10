import argparse
import glob
import os
import random

from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate FID and inception score")
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--ncol', type=int, default=3, required=True)
    parser.add_argument('--nrow', type=int, default=3, required=True)
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    selected_files = random.choices(files, k=args.ncol * args.nrow)

    images = []
    for file_path in selected_files:
        images.append(to_tensor(Image.open(file_path)))
    save_image(images, 'grid.png', nrow=args.nrow)
