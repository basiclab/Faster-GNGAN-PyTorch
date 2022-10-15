import io
import os
from glob import glob

import click
import lmdb
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path, max_readers=1, readonly=True, lock=False,
            readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2    # image & label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            image = txn.get(f'd{idx}'.encode())
            label = txn.get(f'l{idx}'.encode())
            assert image is not None, f"index={idx}"
            assert label is not None, f"index={idx}"
        # decode image
        buf = io.BytesIO()
        buf.write(image)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')

        # decode class
        label = int(label.decode())
        return image, label


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,          # Path to the dataset.
        hflip: bool,        # Horizontal flip augmentation.
        resolution: int,    # Resolution of the images.
        cr: bool,           # Consistency regularization.
    ):
        self.cr = cr
        self.transform_cr = T.Compose([
            T.Resize((resolution, resolution)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(0, translate=(0.2, 0.2)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.RandomHorizontalFlip(p=0.5) if hflip else T.Compose([]),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

        if 'cifar10' in path:
            self.dataset = torchvision.datasets.CIFAR10(
                root=path,
                train=True,
                download=True,
            )
        elif 'stl10' in path:
            self.dataset = torchvision.datasets.STL10(
                root=path,
                split='unlabeled',
                download=True,
            )
        elif 'lsun' in path:
            self.dataset = torchvision.datasets.LSUNClass(
                root=path,
                target_transform=lambda x: 0,
            )
        else:
            self.dataset = LMDBDataset(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        image_ori = self.transform(image)
        if self.cr:
            image_aug = self.transform_cr(image)
        else:
            image_aug = -1
        return image_ori, label, image_aug


@click.command()
@click.option('-d', '--path', type=str, required=True, help='the path to the root directory of dataset.')
@click.option('-o', '--output', type=str, required=True, help='Path to the output lmdb.')
def files_to_lmdb(path, output):
    exts = ["JPEG", "jpg", "PNG", "png"]
    files_in_subdir = []
    files_in_curdir = []
    for ext in exts:
        files_in_subdir.extend(glob(os.path.join(path, f'*/*.{ext}')))
        files_in_curdir.extend(glob(os.path.join(path, f'*.{ext}')))
    files_in_subdir = sorted(files_in_subdir)
    files_in_curdir = sorted(files_in_curdir)
    if len(files_in_subdir) > 0:
        # Each directory is a class.
        subdirs = set(os.path.dirname(f) for f in files_in_subdir)
        dir2cls = {d: i for i, d in enumerate(subdirs)}
        files = files_in_subdir
    else:
        # All files are belonging to the same class.
        assert len(files_in_curdir) > 0, 'No files found in the specified path.'
        dir2cls = {os.path.dirname(files_in_curdir[0]): 0}
        files = files_in_curdir
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with lmdb.open(output, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            for i, file in enumerate(tqdm(files, ncols=0)):
                dir = os.path.dirname(file)
                dkey = f'd{i}'.encode()
                lkey = f'l{i}'.encode()
                # image to bytes
                img = open(file, 'rb').read()
                # class to bytes
                cls = str(dir2cls[dir]).encode()
                # write to db
                txn.put(dkey, img)
                txn.put(lkey, cls)


if __name__ == '__main__':
    files_to_lmdb()
