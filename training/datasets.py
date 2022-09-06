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


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,           # Path to the dataset.
        hflip,          # Horizontal flip augmentation.
        resolution,     # Resolution of the images.
        apply_cr,       # Consistency regularization gamma.
    ):
        self.apply_cr = apply_cr
        if self.apply_cr:
            self.cr_transform = T.Compose([
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

        self.env = lmdb.open(
            path, max_readers=32, readonly=True, lock=False, readahead=False,
            meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2    # #image + #label

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img = txn.get(f'd{index}'.encode())
            cls = txn.get(f'l{index}'.encode())
            assert img is not None, f"index={index}"
            assert cls is not None, f"index={index}"
        # decode image
        buf = io.BytesIO()
        buf.write(img)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # decode class
        cls = int(cls.decode())

        img_ori = self.transform(img)
        if self.apply_cr:
            img_aug = self.cr_transform(img)
        else:
            img_aug = -1
        return img_ori, cls, img_aug


@click.command()
@click.option('-d', '--dataset', type=str, required=True,
              help='If `--dataset cifar10` or `--dataset stl10` is used, '
                   'the dataset will be automatically downloaded. Otherwise, '
                   'the path to the root of dataset must be specified.')
@click.option('-o', '--output', type=str, required=True,
              help='Path to the output lmdb.')
def files_to_lmdb(dataset, output):

    if dataset in ['cifar10', 'stl10', 'church', 'bedroom', 'horse', 'cat']:
        if dataset == 'cifar10':
            pt_dataset = torchvision.datasets.CIFAR10(
                root='/tmp', train=True, download=True)
        elif dataset == 'stl10':
            pt_dataset = torchvision.datasets.STL10(
                root='/tmp', split='unlabeled', download=True)
        else:
            pt_dataset = torchvision.datasets.LSUNClass(
                os.path.join('./data/lsun', dataset),
                target_transform=lambda x: 0)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with lmdb.open(output, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                for i, (img, y) in enumerate(tqdm(pt_dataset, ncols=0)):
                    dkey = f'd{i}'.encode()
                    lkey = f'l{i}'.encode()
                    # image to bytes
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    img = buf.getvalue()
                    # class to bytes
                    cls = str(y).encode()
                    # write to db
                    txn.put(dkey, img)
                    txn.put(lkey, cls)
    else:
        exts = ["JPEG", "jpg", "PNG", "png"]
        files_in_subdir = []
        files_in_curdir = []
        for ext in exts:
            files_in_subdir.extend(glob(os.path.join(dataset, f'*/*.{ext}')))
            files_in_curdir.extend(glob(os.path.join(dataset, f'*.{ext}')))
        files_in_subdir = sorted(files_in_subdir)
        files_in_curdir = sorted(files_in_curdir)
        if len(files_in_subdir) > 0:
            # Each directory is a class.
            subdirs = set(os.path.basename(os.path.dirname(f)) for f in files_in_subdir)
            dir2cls = {d: i for i, d in enumerate(subdirs)}
            files = files_in_subdir
        else:
            # All files are belonging to the same class.
            assert len(files_in_curdir) > 0, 'No files found in the specified path.'
            dir2cls = {os.path.basename(os.path.dirname(files_in_curdir[0])): 0}
            files = files_in_curdir
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with lmdb.open(output, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                for i, file in enumerate(tqdm(files, ncols=0)):
                    dir = os.path.basename(os.path.dirname(file))
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
