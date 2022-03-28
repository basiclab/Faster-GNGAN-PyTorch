import io

import lmdb
from PIL import Image
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets import VisionDataset


class LMDBDataset(VisionDataset):
    def __init__(self, path, transform):
        self.env = lmdb.open(path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            imgbytes = txn.get(f'{index}'.encode())

        buf = io.BytesIO()
        buf.write(imgbytes)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def get_dataset(name):
    """Get datasets

    Args:
        name: the format [name].[resolution],
              i.g., cifar10.32, celebahq.256
    """
    name, img_size = name.split('.')
    img_size = int(img_size)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    dataset = None
    if name == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
    if name == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True, transform=transform)
    if name == 'celebahq':
        dataset = LMDBDataset(
            f'./data/celebahq/{img_size}', transform=transform)
    if name == 'lsun_church':
        dataset = datasets.LSUNClass(
            './data/lsun/church/', transform, (lambda x: 0))
    if name == 'lsun_bedroom':
        dataset = datasets.LSUNClass(
            './data/lsun/bedroom', transform, (lambda x: 0))
    if name == 'lsun_horse':
        dataset = datasets.LSUNClass(
            './data/lsun/horse', transform, (lambda x: 0))
    if name == 'imagenet':
        dataset = LMDBDataset(
            './data/imagenet/train', transform=transform)
    if dataset is None:
        raise ValueError(f'Unknown dataset {name}')
    return dataset


if __name__ == '__main__':
    import argparse
    import os
    from glob import glob
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            files = []
            files.extend(
                glob(os.path.join(args.path, '**/*.jpg'), recursive=True))
            files.extend(
                glob(os.path.join(args.path, '**/*.JPEG'), recursive=True))
            files.extend(
                glob(os.path.join(args.path, '**/*.png'), recursive=True))
            try:
                files = sorted(
                    files,
                    key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
                )
                print("Sort by file number")
            except ValueError:
                files = sorted(files)
                print("Sort by file path")
            for i, file in enumerate(tqdm(files, dynamic_ncols=True)):
                key = f'{i}'.encode()
                img = open(file, 'rb').read()
                txn.put(key, img)
