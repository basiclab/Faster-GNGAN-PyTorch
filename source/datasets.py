import os

import h5py as h5
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(self, name, img_size, transform, in_memory=True,
                 cache='./data', chunk_size=500):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, '%s.%s.hdf5' % (name, img_size))
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(self.h5_path, name, chunk_size, img_size)

        if in_memory:
            with h5.File(self.h5_path, 'r') as f:
                self.images = f['images'][:]
                self.labels = f['labels'][:]
            self.num_images = len(self.images)
        else:
            self.num_images = len(h5.File(self.h5_path, 'r')['images'])

    def create_hdf5(self, h5_path, name, chunk_size, img_size):
        dataset_name = '%s.%s.raw' % (name, img_size)
        dataloader = DataLoader(
            dataset=get_dataset(
                dataset_name, aug_transform=transforms.Compose([])),
            batch_size=50, num_workers=8)
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, 'w') as f:
            print('Dataset size: %d' % len(dataloader.dataset))
            imgs_dset = f.create_dataset(
                'images', (0, 3, img_size, img_size), dtype='uint8',
                maxshape=(len(dataloader.dataset), 3, img_size, img_size),
                chunks=(chunk_size, 3, img_size, img_size))
            print('Image chunks chosen as:', imgs_dset.chunks)
            # imgs_dset[...] = x
            labels_dset = f.create_dataset(
                'labels', (0,), dtype='int64',
                maxshape=(len(dataloader.dataset),),
                chunks=(chunk_size,))
            print('Label chunks chosen as:', labels_dset.chunks)
            # labels_dset[...] = y
        with tqdm(dataloader, dynamic_ncols=True) as pbar:
            for i, (x, y) in enumerate(pbar):
                x = (x + 1) / 2                 # [0., 1.]
                x = (x * 255).byte().numpy()    # [0, 255]
                y = y.numpy()
                with h5.File(h5_path, 'a') as f:
                    f['images'].resize(
                        f['images'].shape[0] + x.shape[0], axis=0)
                    f['images'][-x.shape[0]:] = x
                    f['labels'].resize(
                        f['labels'].shape[0] + y.shape[0], axis=0)
                    f['labels'][-y.shape[0]:] = y

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
        image = self.transform(torch.tensor(image))
        return image, label


class CenterCropLongEdge(object):
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))


def get_dataset(name,
                aug_transform=transforms.RandomHorizontalFlip(),
                in_memory=True):
    """Get datasets

    Args:
        name: in format [name].[resolution: int].[format: "hdf5" or "raw"],
              i.g., cifar10.32.raw, imagenet.128.hdf5
        in_memory: if the type of format is hdf5, in_memory controls that the
                   dataset is loaded in RAM or not.
    """
    name, img_size, hdf5 = name.split('.')
    img_size = int(img_size)
    assert name in [
        'cifar10', 'stl10', 'imagenet', 'celebhq', 'lsun_church_outdoor']
    assert hdf5 in ['raw', 'hdf5']

    if name in ['cifar10', 'stl10']:
        # these datasets are small enough to load in memory
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            aug_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if name == 'cifar10':
            return torchvision.datasets.CIFAR10(
                root='./data', train=True,
                download=True, transform=transform)
        if name == 'stl10':
            return torchvision.datasets.STL10(
                './data', split='unlabeled',
                download=True, transform=transform)

    if hdf5 == 'raw':
        transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize((img_size, img_size)),
            aug_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # raw
        if name == 'imagenet':
            return torchvision.datasets.ImageFolder(
                './data/imagenet/train', transform=transform)
        if name == 'celebhq':
            return torchvision.datasets.ImageFolder(
                './data/celebhq/', transform=transform)
        if name == 'lsun_church_outdoor':
            return torchvision.datasets.LSUN(
                './data/lsun/',
                classes=['church_outdoor_train'], transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            aug_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # hdf5
        return HDF5Dataset(name, img_size, transform, in_memory=in_memory)


if __name__ == "__main__":
    dataset = get_dataset(
        'celebhq.256.hdf5', aug_transform=transforms.Compose([]))
    print(len(dataset))
    image, label = dataset[0]
    print('image', image.shape, image.dtype, image.min(), image.max())
    print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'celebhq.png')

    dataset = get_dataset(
        'lsun_church_outdoor.256.hdf5', aug_transform=transforms.Compose([]))
    print(len(dataset))
    image, label = dataset[0]
    print('image', image.shape, image.dtype, image.min(), image.max())
    print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'lsun_church_outdoor.png')

    # dataset = get_dataset(
    #     'imagenet.128.hdf5', aug_transform=transforms.Compose([]))
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)
    # torchvision.utils.save_image((image + 1) / 2, 'imagenet.png')
