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
        img = Image.open(buf)

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def get_dataset(name, in_memory=True):
    """Get datasets

    Args:
        name: the format [name].[resolution],
              i.g., cifar10.32, celebahq.256
        in_memory: load dataset into memory.
    """
    name, img_size = name.split('.')
    img_size = int(img_size)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        dataset = datasets.ImageFolder(
            './data/imagenet/raw/train', transform, (lambda x: 0))
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
            files = glob(os.path.join(args.path, '**/*.JPEG'))
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

2021-12-09 20:53:55.959640: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-12-09 20:53:57.313880: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-12-09 20:53:57.438473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:1b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-12-09 20:53:57.439942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 1 with properties: 
pciBusID: 0000:3d:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-12-09 20:53:57.439991: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-12-09 20:53:57.441829: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-12-09 20:53:57.443890: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-12-09 20:53:57.444224: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-12-09 20:53:57.446118: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-12-09 20:53:57.447247: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-12-09 20:53:57.450888: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-12-09 20:53:57.456482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0, 1
2021-12-09 20:53:57.456825: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-09 20:53:57.462580: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3000000000 Hz
2021-12-09 20:53:57.463388: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x43d91f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-12-09 20:53:57.463414: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-12-09 20:53:57.688940: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x61723c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-12-09 20:53:57.689007: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0
2021-12-09 20:53:57.689031: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (1): Tesla V100-SXM2-32GB, Compute Capability 7.0
2021-12-09 20:53:57.694009: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:1b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-12-09 20:53:57.696342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 1 with properties: 
pciBusID: 0000:3d:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
2021-12-09 20:53:57.696389: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-12-09 20:53:57.696409: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-12-09 20:53:57.696420: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-12-09 20:53:57.696431: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-12-09 20:53:57.696442: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-12-09 20:53:57.696453: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-12-09 20:53:57.696464: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-12-09 20:53:57.701905: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0, 1
2021-12-09 20:53:57.701952: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-12-09 20:53:58.567801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-12-09 20:53:58.567851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 1 
2021-12-09 20:53:58.567871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N Y 
2021-12-09 20:53:58.567880: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 1:   Y N 
2021-12-09 20:53:58.572277: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30132 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:1b:00.0, compute capability: 7.0)
2021-12-09 20:53:58.574284: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 30132 MB memory) -> physical GPU (device: 1, name: Tesla V100-SXM2-32GB, pci bus id: 0000:3d:00.0, compute capability: 7.0)
WARNING:tensorflow:From /home/u0000627/GEgan/keras_gcnn/layers/normalization.py:192: calling Layer.add_update (from tensorflow.python.keras.engine.base_layer) with inputs is deprecated and will be removed in a future version.
Instructions for updating:
inputs is now automatically inferred
2021-12-09 20:54:01.541453: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-12-09 20:54:01.776898: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
evaluating images with  100  size...
Traceback (most recent call last):
  File "train.py", line 333, in <module>
    (IS, IS_std), FID = get_inception_score_and_fid(images, './stats/cifar10.train.npz',use_torch=False)
  File "/home/u0000627/.local/lib/python3.6/site-packages/pytorch_gan_metrics/utils.py", line 61, in get_inception_score_and_fid
    images, dims=[2048, 1008], use_torch=use_torch, **kwargs)
  File "/home/u0000627/.local/lib/python3.6/site-packages/pytorch_gan_metrics/core.py", line 81, in get_inception_feature
    outputs = model(batch_images)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/u0000627/.local/lib/python3.6/site-packages/pytorch_gan_metrics/inception.py", line 158, in forward
    x = block(x)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/container.py", line 119, in forward
    input = module(input)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torchvision/models/inception.py", line 477, in forward
    x = self.bn(x)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py", line 140, in forward
    self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
  File "/home/u0000627/.local/lib/python3.6/site-packages/torch/nn/functional.py", line 2150, in batch_norm
    input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled
RuntimeError: CUDA out of memory. Tried to allocate 132.00 MiB (GPU 0; 31.75 GiB total capacity; 413.30 MiB already allocated; 91.75 MiB free; 422.00 MiB reserved in total by PyTorch)