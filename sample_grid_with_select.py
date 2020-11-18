import argparse
import glob
import os
import random

import numpy as np
from PIL import Image
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate FID and inception score")
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--source_image', type=str,
                        default='./figures/grid.png')
    parser.add_argument('--ncol', type=int, default=3)
    parser.add_argument('--nrow', type=int, default=3)
    parser.add_argument('--shift_i', type=int, default=0)
    parser.add_argument('--shift_j', type=int, default=0)
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    transform = Compose([Resize((args.size, args.size)), ToTensor()])

    try:
        image = to_tensor(Image.open(args.source_image))
        print('Read source image')
    except Exception:
        random_files = np.random.choice(
            files, args.ncol * args.nrow, replace=False)
        images = []
        for path in random_files:
            image = Image.open(path)
            images.append(transform(image))
        image = make_grid(images, nrow=args.nrow, padding=0)
        args.shift_i = 0
        args.shift_j = 0
        print('Randomly initialize image')
    save_image(image, args.source_image)

    file_index = 0
    now_i, now_j = 0, 0
    os.makedirs('figures', exist_ok=True)
    while True:
        image = to_tensor(Image.open(args.source_image))

        command = input('Current position (%d, %d):' % (now_i, now_j))

        if len(command) != 0 and command[0] == 'i':     # set index
            now_i, now_j = [int(x) for x in command[1:].split()]
            now_i = now_i + args.shift_i
            now_j = now_j + args.shift_j
            continue

        elif len(command) != 0 and command[0] == 'r':     # random all images
            random_files = np.random.choice(
                files, args.ncol * args.nrow, replace=False)
            images = []
            for path in random_files:
                images.append(transform(Image.open(path)))
            patch = make_grid(images, nrow=args.nrow, padding=0)
            image[
                :,
                args.shift_i * args.size:
                    args.shift_i * args.size + patch.shape[1],
                args.shift_j * args.size:
                    args.shift_j * args.size + patch.shape[2]] = \
                patch

        elif len(command) != 0:
            if command[0] == 'R':     # random image index
                file_index = random.randint(0, len(files) - 1)

            if command[0] == ']':     # apply and go to next
                file_index = (file_index + 1) % len(files)

            if command[0] == '[':     # apply and go to prev
                file_index = (file_index - 1 + len(files)) % len(files)

            new_image = to_tensor(Image.open(files[file_index]))
            image[
                :,
                now_i * args.size: (now_i + 1) * args.size,
                now_j * args.size: (now_j + 1) * args.size] = \
                new_image

        save_image(image, args.source_image)
