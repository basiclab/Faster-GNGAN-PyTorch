import io

import lmdb
import torch
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, resolution, hflip):
        transform = [transforms.Resize((resolution, resolution))]
        if hflip:
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

        self.env = lmdb.open(
            path, max_readers=1, readonly=True, lock=False, readahead=False,
            meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2    # image + label

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

        img = self.transform(img)
        return img, cls


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


if __name__ == '__main__':
    import argparse
    import os
    from glob import glob
    from tqdm import tqdm

    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=['cifar10', 'stl10'], default=None)
    parser.add_argument(
        '--path', type=str, default=None)
    parser.add_argument(
        '--out', type=str, required=True)
    args = parser.parse_args()

    if args.dataset is None and args.path is None:
        print('one of --dataset or --path must be specified')

    if args.dataset:
        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root='/tmp', train=True, download=True)
        if args.dataset == 'stl10':
            dataset = torchvision.datasets.STL10(
                root='/tmp', split='unlabeled', download=True)
        with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                for i, (img, y) in enumerate(tqdm(dataset, ncols=0)):
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
        with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                files = []
                files.extend(
                    glob(os.path.join(args.path, '**/*.jpg'), recursive=True))
                files.extend(
                    glob(os.path.join(args.path, '**/*.JPEG'), recursive=True))
                files.extend(
                    glob(os.path.join(args.path, '**/*.png'), recursive=True))
                files = sorted(files)
                for i, file in enumerate(tqdm(files, ncols=0)):
                    dkey = f'd{i}'.encode()
                    lkey = f'l{i}'.encode()
                    # image to bytes
                    img = open(file, 'rb').read()
                    # class to bytes
                    cls = "0".encode()
                    # write to db
                    txn.put(dkey, img)
                    txn.put(lkey, cls)
