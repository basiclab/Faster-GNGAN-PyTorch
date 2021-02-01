# Gradient Normalization for Generative Adversarial Networks

The author's official implementation of GN-GAN.

## Setup & Requirements
- Clone project
    ```
    git clone https://github.com/w86763777/pytorch-gngan.git
    cd pytorch-gngan
    git submodule init
    git submodule update
    ```
- We use python 3.6
- Install python packages
    ```bash
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10

    We use Pytorch build-in CIFAR-10.

- STL-10

    We use Pytorch build-in STL-10.

- CelebA-HQ 128x128

    Please manually split dataset into 27k for training and 3k for testing. The
    Folder structure is as follows:
    ```
    data/celebhq
    ├── train128
    │   └── dummy
    │       ├── 00001.jpg
    │       ├── ...
    └── val128
        └── dummy
            ├── 27001.jpg
            ├── ...
    ```

- [LSUN Church Outdoor](https://www.yf.io/p/lsun) 128x128

    Folder structure is as follows:
    ```
    data/lsun/
    ├── church_outdoor_train_lmdb
    │   ├── data.mdb
    │   └── lock.mdb
    └── church_outdoor_val_lmdb
        ├── data.mdb
        └── lock.mdb
    ```

## How to Run
- There are 3 training scripts in `gans`, i.e., `train_gan.py`, `train_cgan.py` and `train_gan128.py`. Following is the Table of compatible configuration.

    |script         |configurations|
    |---------------|--------------|
    |train_gan.py   |`GN-GAN_CIFAR10_CNN.txt`,<br>`GN-GAN_CIFAR10_RES.txt`,<br>`GN-GAN_STL10_CNN.txt`,<br>`GN-GAN_STL10_RES.txt`,<br>`GN-GAN-CR_CIFAR10_CNN.txt`,<br>`GN-GAN-CR_CIFAR10_RES.txt`,<br>`GN-GAN-CR_STL10_CNN.txt`,<br>`GN-GAN-CR_STL10_RES.txt`|
    |train_cgan.py  |`GN-cGAN_CIFAR10_BIGGAN.txt`|
    |train_gan128.py|`GN-GAN_CELEBHQ128_RES.txt`,<br>`GN-GAN_CHURCH128_RES.txt`|

- Run the script with the compatible configuration, i.e.,
    ```
    python gans/train_gan.py \
        --flagfile ./config/GN-GAN-CR_CIFAR10_RES.txt \
        --logdir ./logs/GN-GAN-CR_CIFAR10_RES
    ```

- Generate images from pretrained model.

    For example,
    ```
    python gans/train_gan.py \
        --flagfile ./logs/GN-GAN-CR_CIFAR10_RES/flagfile.txt \
        --generate \
        --num_images 50000
    ```

    The generated samples are saved into `./logs/GN-GAN-CR_CIFAR10_RES/generate`


## Samples
- GN-GAN-CR_CIFAR10_RES.txt

    ![](./figures/cifar10_res_cr.png)
- GN-cGAN_CIFAR10_BIGGAN.txt

    ![](./figures/cifar10_biggan_10x10.png)
- GN-GAN_CHURCH128_RES.txt

    ![](./figures/lsun_church128_3x3.png)
- GN-GAN_CELEBHQ128_RES.txt

    ![](./figures/celebhq128_3x3.png)