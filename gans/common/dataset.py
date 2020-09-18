import os

import h5py as h5
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, ToPILImage
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
class ImageNet(Dataset):
    def __init__(self, root, transform=None,
                 cache='./data', chunk_size=500, size=128, in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.h5_path = os.path.join(cache, 'imagenet%d.hdf5' % size)
        if not os.path.isfile(self.h5_path):
            self.create_hdf5(root, self.h5_path, chunk_size, size)

        if in_memory:
            with h5.File(self.h5_path, 'r') as f:
                self.images = f['images'][:]
                self.labels = f['labels'][:]

    def create_hdf5(self, root, h5_path, chunk_size, size):
        dataloader = DataLoader(
            dataset=ImageFolder(
                root=root,
                transform=Compose([Resize((size, size)), ToTensor()])),
            batch_size=50, num_workers=8)
        with h5.File(h5_path, 'w') as f:
            print('Dataset size: %d' % len(dataloader.dataset))
            imgs_dset = f.create_dataset(
                'images', (0, 3, size, size), dtype='uint8',
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
                x = (x * 255).byte().numpy()
                y = y.numpy()
                with h5.File(h5_path, 'a') as f:
                    f['images'].resize(
                        f['images'].shape[0] + x.shape[0], axis=0)
                    f['images'][-x.shape[0]:] = x
                    f['labels'].resize(
                        f['labels'].shape[0] + y.shape[0], axis=0)
                    f['labels'][-y.shape[0]:] = y

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
            label = self.labels[idx]
        else:
            with h5.File(self.h5_path, 'r') as f:
                image = f['images'][idx]
                label = f['labels'][idx]
        image = ToPILImage()(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    dataset = ImageNet('./data/ILSVRC2012/train')
