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
                T.ToTensor(),
                T.Lambda(lambda x: torch.clamp(
                    x + torch.randint_like(x, 0, 2) / 255, min=0, max=1
                )),
                T.RandomHorizontalFlip(),
                T.RandomAffine(0, translate=(0.2, 0.2)),
            ])
        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.RandomHorizontalFlip(p=0.5) if hflip else T.Compose([]),
            T.ToTensor(),
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
        # decode image
        buf = io.BytesIO()
        buf.write(img)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # decode class
        cls = int(cls.decode())

        if self.apply_cr is not None:
            img_aug = self.cr_transform(img)
        else:
            img_aug = None
        img = self.transform(img)
        return img, cls, img_aug


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(len(self.dataset), generator=g).tolist()
            window = int(len(order) * self.window_size)
        else:
            order = torch.arange(len(self.dataset)).tolist()
            window = 0

        idx = 0
        while True:
            i = idx % len(order)
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - torch.randint(window, [], generator=g)) % len(order)
                order[i], order[j] = order[j], order[i]
            idx += 1


@click.command()
@click.option('-d', '--dataset', type=str, required=True,
              help='If `--dataset cifar10` or `--dataset stl10` is used, '
                   'the dataset will be automatically downloaded. Otherwise, '
                   'the path to the root of dataset must be specified.')
@click.option('-o', '--output', type=str, required=True,
              help='Path to the output lmdb.')
def files_to_lmdb(dataset, output):

    if dataset in ['cifar10', 'stl10']:
        if dataset == 'cifar10':
            pt_dataset = torchvision.datasets.CIFAR10(
                root='/tmp', train=True, download=True)
        if dataset == 'stl10':
            pt_dataset = torchvision.datasets.STL10(
                root='/tmp', split='unlabeled', download=True)
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
            subdirs = set(os.path.basename(os.path.dirname(f)) for f in files_in_subdir)
            dir2cls = {d: i for i, d in enumerate(subdirs)}
            files = files_in_subdir
        else:
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
