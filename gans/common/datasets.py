import os

import h5py as h5
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# HDF5 supports chunking and compression. You may want to experiment
# with different chunk sizes to see how it runs on your machines.
# Chunk Size/compression    |Read speed      |Read speed|Filesize|Time to write
#                           |@ 256x256       |@128x128  |@128x128|@128x128
# --------------------------|----------------|----------|--------|-------------
# 1 / None                  |20/s            |          |        |
# 500 / None                |ramps up to 77/s|102/s     |61GB    |23min
# 500 / LZF                 |                |8/s       |56GB    |23min
# 1000 / None               |78/s            |          |        |
# 5000 / None               |81/s            |          |        |
# auto:(125,1,16,32) / None |11/s            |          |61GB    |
class ImageNetHDF5(Dataset):
    def __init__(self, root, transform=None,
                 cache='./data', chunk_size=500, size=128, in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, 'imagenet%d.hdf5' % size)
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(root, self.h5_path, chunk_size, size)

        if in_memory:
            with h5.File(self.h5_path, 'r') as f:
                self.images = f['imgs'][:]
                self.labels = f['labels'][:]
            self.num_images = len(self.labels)
        else:
            self.num_images = len(h5.File(self.h5_path, 'r')['labels'])

    def create_hdf5(self, root, h5_path, chunk_size, size):
        dataloader = DataLoader(
            dataset=get_dataset('imagenet128'), batch_size=50, num_workers=8)
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, 'w') as f:
            print('Dataset size: %d' % len(dataloader.dataset))
            imgs_dset = f.create_dataset(
                'imgs', (0, 3, size, size), dtype='uint8',
                maxshape=(len(dataloader.dataset), 3, size, size),
                chunks=(chunk_size, 3, size, size))
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
                    f['imgs'].resize(
                        f['imgs'].shape[0] + x.shape[0], axis=0)
                    f['imgs'][-x.shape[0]:] = x
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
                image = f['imgs'][idx]
                label = f['labels'][idx]
        if self.transform is not None:
            image = transforms.functional.to_pil_image(torch.tensor(image))
            image = self.transform(image)
        else:
            image = torch.tensor(image).float() / 255 * 2 - 1
        return image, label


class CelebHQ128HDF5(Dataset):
    def __init__(self, transform=None, cache='./data', chunk_size=500,
                 in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, 'celebhq128.hdf5')
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(self.h5_path, chunk_size, 128)

        if in_memory:
            with h5.File(self.h5_path, 'r') as f:
                self.images = f['imgs'][:]
            self.num_images = len(self.images)
        else:
            self.num_images = len(h5.File(self.h5_path, 'r')['imgs'])

    def create_hdf5(self, h5_path, chunk_size, size):
        dataloader = DataLoader(
            dataset=get_dataset('celebhq128'), batch_size=50, num_workers=8)
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, 'w') as f:
            print('Dataset size: %d' % len(dataloader.dataset))
            imgs_dset = f.create_dataset(
                'imgs', (0, 3, size, size), dtype='uint8',
                maxshape=(len(dataloader.dataset), 3, size, size),
                chunks=(chunk_size, 3, size, size))
            print('Image chunks chosen as:', imgs_dset.chunks)
        with tqdm(dataloader, dynamic_ncols=True) as pbar:
            for i, (x, y) in enumerate(pbar):
                x = (x + 1) / 2                 # [0., 1.]
                x = (x * 255).byte().numpy()    # [0, 255]
                y = y.numpy()
                with h5.File(h5_path, 'a') as f:
                    f['imgs'].resize(
                        f['imgs'].shape[0] + x.shape[0], axis=0)
                    f['imgs'][-x.shape[0]:] = x

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
        else:
            with h5.File(self.h5_path, 'r') as f:
                image = f['imgs'][idx]
        if self.transform is not None:
            image = transforms.functional.to_pil_image(torch.tensor(image))
            image = self.transform(image)
        else:
            image = torch.tensor(image).float() / 255 * 2 - 1
        return image, torch.zeros((1,)).long()


class LSUNChurchOutdoor128HDF5(Dataset):
    def __init__(self, transform=None, cache='./data', chunk_size=500,
                 in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, 'lsun_church_outdoor.hdf5')
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(self.h5_path, chunk_size, 128)

        if in_memory:
            with h5.File(self.h5_path, 'r') as f:
                self.images = f['imgs'][:]
            self.num_images = len(self.images)
        else:
            self.num_images = len(h5.File(self.h5_path, 'r')['imgs'])

    def create_hdf5(self, h5_path, chunk_size, size):
        dataloader = DataLoader(
            dataset=get_dataset('lsun_church_outdoor'), batch_size=50,
            num_workers=8)
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        with h5.File(h5_path, 'w') as f:
            print('Dataset size: %d' % len(dataloader.dataset))
            imgs_dset = f.create_dataset(
                'imgs', (0, 3, size, size), dtype='uint8',
                maxshape=(len(dataloader.dataset), 3, size, size),
                chunks=(chunk_size, 3, size, size))
            print('Image chunks chosen as:', imgs_dset.chunks)
        with tqdm(dataloader, dynamic_ncols=True) as pbar:
            for i, (x, y) in enumerate(pbar):
                x = (x + 1) / 2                 # [0., 1.]
                x = (x * 255).byte().numpy()    # [0, 255]
                y = y.numpy()
                with h5.File(h5_path, 'a') as f:
                    f['imgs'].resize(
                        f['imgs'].shape[0] + x.shape[0], axis=0)
                    f['imgs'][-x.shape[0]:] = x

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
        else:
            with h5.File(self.h5_path, 'r') as f:
                image = f['imgs'][idx]
        if self.transform is not None:
            image = transforms.functional.to_pil_image(torch.tensor(image))
            image = self.transform(image)
        else:
            image = torch.tensor(image).float() / 255 * 2 - 1
        return image, torch.zeros((1,)).long()


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def get_dataset(name):
    assert name in [
        'cifar10', 'stl10',
        'imagenet128', 'imagenet128.hdf5',
        'celebhq128', 'celebhq128.hdf5',
        'lsun_church_outdoor', 'lsun_church_outdoor.hdf5']
    if name == 'cifar10':
        return torchvision.datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if name == 'stl10':
        return torchvision.datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if name == 'imagenet128':
        return torchvision.datasets.ImageFolder(
            './data/imagenet/train',
            transform=transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if name == 'imagenet128.hdf5':
        return ImageNetHDF5(size=128, in_memory=True)
    if name == 'celebhq128':
        return torchvision.datasets.ImageFolder(
            './data/celebhq/train128/',
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if name == 'celebhq128.hdf5':
        return CelebHQ128HDF5(
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]), in_memory=True)
    if name == 'lsun_church_outdoor':
        return torchvision.datasets.LSUN(
            './data/lsun/',
            classes=['church_outdoor_train'],
            transform=transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if name == 'lsun_church_outdoor.hdf5':
        return LSUNChurchOutdoor128HDF5(
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))


if __name__ == "__main__":
    # Create cache for imagenet128
    # dataset = get_dataset('imagenet128.hdf5')
    # print(len(dataset))
    # image, label = dataset[0]
    # print('image', image.shape, image.dtype, image.min(), image.max())
    # print('label', label.shape, label.dtype)

    # Create cache for celebhq128
    dataset = get_dataset('celebhq128.hdf5')
    print(len(dataset))
    image, label = dataset[0]
    print('image', image.shape, image.dtype, image.min(), image.max())
    print('label', label.shape, label.dtype)

    # Create cache for lsun church_outdoor
    dataset = get_dataset('lsun_church_outdoor.hdf5')
    print(len(dataset))
    image, label = dataset[0]
    print('image', image.shape, image.dtype, image.min(), image.max())
    print('label', label.shape, label.dtype)
