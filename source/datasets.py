import os

import h5py as h5
import torchvision
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(self, dataset, name, img_size, transform=None, in_memory=True,
                 cache='./data', compression=None):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, '%s.%s.hdf5' % (name, img_size))
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(dataset, self.h5_path, compression)

        with h5.File(self.h5_path, 'r') as f:
            if in_memory:
                self.images = f['images'][:]
                self.labels = f['labels'][:]
                self.num_images = len(self.images)
            else:
                self.num_images = len(f['images'])

    def create_hdf5(self, dataset, h5_path, compression):
        dataloader = DataLoader(dataset, batch_size=50, num_workers=8)
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, 'w') as f:
            shape = dataset[0][0].shape
            f.create_dataset(
                'images', shape=(0, *shape), dtype='uint8',
                maxshape=(len(dataset), *shape), compression=compression)
            f.create_dataset(
                'labels', shape=(0,), dtype='int64',
                maxshape=(len(dataset),), compression=compression)
        for x, y in tqdm(dataloader, dynamic_ncols=True, leave=False,
                         desc='create hdf5'):
            x = (x * 255).byte().numpy()    # [0, 255]
            y = y.long().numpy()
            with h5.File(h5_path, 'a') as f:
                f['images'].resize(
                    f['images'].shape[0] + x.shape[0], axis=0)
                f['images'][-x.shape[0]:] = x
                f['labels'].resize(
                    f['labels'].shape[0] + y.shape[0], axis=0)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
            label = self.labels[idx]
        else:
            with h5.File(self.h5_path, 'r') as f:
                image = f['images'][idx]
                label = f['labels'][idx]
        image = to_pil_image(torch.tensor(image))
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataset(name, in_memory=True):
    """Get datasets

    Args:
        name: the format [name].[resolution].["hdf5"|"raw"],
              i.g., cifar10.32.raw, imagenet.128.hdf5
        in_memory: if the type of format is hdf5, in_memory controls that the
                   dataset is loaded in RAM or not.
    """
    name, img_size, fmt = name.split('.')
    img_size = int(img_size)

    assert fmt in ['raw', 'hdf5']
    if fmt == 'raw':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if fmt == 'hdf5':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        transform_hdf5 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    target_transform = (lambda x: 0)

    assert name in [
        'cifar10', 'stl10', 'imagenet', 'celebhq', 'celebhq_train', 'ffhq',
        'lsun_church', 'lsun_bedroom', 'lsun_horse']
    if name == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
    if name == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True, transform=transform)
    if name == 'imagenet':
        dataset = datasets.ImageFolder(
            './data/imagenet/train', transform=transform)
    if name == 'celebhq':
        dataset = datasets.ImageFolder(
            f'./data/celebhq/{img_size}', transform=transform)
    if name == 'celebhq_train':
        dataset = datasets.ImageFolder(
            f'./data/celebhq/{img_size}train', transform=transform)
    if name == 'ffhq':
        dataset = datasets.ImageFolder(
            './data/ffhq/', transform=transform)
    if name == 'lsun_church':
        dataset = datasets.LSUNClass(
            './data/lsun/church/', transform, target_transform)
    if name == 'lsun_bedroom':
        dataset = datasets.LSUNClass(
            './data/lsun/bedroom', transform, target_transform)
    if name == 'lsun_horse':
        dataset = datasets.LSUNClass(
            './data/lsun/horse', transform, target_transform)

    if fmt == 'hdf5':
        dataset = HDF5Dataset(
            dataset, name, img_size, transform_hdf5, in_memory=in_memory)

    return dataset


if __name__ == "__main__":
    # dataset = get_dataset('ffhq.1024.hdf5')
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'ffhq.png')

    dataset = get_dataset('lsun_horse.256.hdf5')
    print(len(dataset))
    image, label = dataset[0]
    print('image', image.shape, image.dtype, image.min(), image.max())
    print('label', label.shape, label.dtype)
    torchvision.utils.save_image((image + 1) / 2, 'lsun_horse.png')

    # dataset = get_dataset(
    #     'celebhq.256.hdf5', aug_transform=transforms.Compose([]))
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'celebhq.png')

    # dataset = get_dataset(
    #     'lsun_church_outdoor.256.hdf5', aug_transform=transforms.Compose([]))
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'lsun_church_outdoor.png')

    # dataset = get_dataset(
    #     'lsun_bedroom.256.hdf5', aug_transform=transforms.Compose([]))
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'lsun_bedroom.png')

    # dataset = get_dataset(
    #     'imagenet.128.hdf5', aug_transform=transforms.Compose([]))
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'imagenet.png')
