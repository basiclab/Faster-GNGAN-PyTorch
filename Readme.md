# Gradient Normalization for Generative Adversarial Networks

The author's official implementation of Gradient Normalized GAN (GN-GAN).

## Recommended System Requirements
- CUDA 10.2
- Python 3.8.9
- Python packages
    ```sh
    # update `pip` for installing latest tensorboard.
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10

    Pytorch build-in CIFAR-10 will be downloaded on the fly.

- STL-10

    Pytorch build-in STL-10 will be downloaded on the fly.

- CelebA-HQ 128/256

    We obtain celeba-hq from [this repository](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download) and preprocess it into `lmdb` file.
    - 256x256
        ```
        python dataset.py path/to/celebahq/256 ./data/celebahq/256
        ```
    - 128x128

        We split data into train test split by filename, the test set contains images from 27001.jpg to 30000.jpg.
        ```
        python dataset.py path/to/celebahq/128/train ./data/celebahq/256
        ```
    Our folder structure:
    ```
    ./data/celebahq
    ├── 128
    │   ├── data.mdb
    │   └── lock.mdb
    └── 256
        ├── data.mdb
        └── lock.mdb
    ```

- LSUN Church Outdoor 256x256

    Our folder structure:
    ```
    ./data/lsun/church/
    ├── data.mdb
    └── lock.mdb
    ```

## Preprocessing Datasets for FID
Pre-calculated statistic for FID can be downloaded [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing):
- cifar10.train.npz - Training set of CIFAR10
- cifar10.test.npz - Testing set of CIFAR10
- stl10.unlabeled.48.npz - Unlabeled set of STL10 in resolution 48x48
- celebahq.3k.128.npz - Last 3k images of CelebA-HQ 128x128
- celebahq.all.256.npz - Full dataset of CelebA-HQ 256x256
- lsun_church.train.256.npz - Training set of LSUN Church Outdoor

Our folder structure:
```
./stats
├── celebahq.3k.128.npz
├── celebahq.all.256.npz
├── church.train.256.npz
├── cifar10.test.npz
├── cifar10.train.npz
└── stl10.unlabeled.48.npz
```

**NOTE**

All the reported value in our paper are calculated by official implementation of Inception Score and FID.


## Training
- Configuration files
    - We use `absl-py` to parse/save/reload the command line arguments.
    - All the configurations can be found in `./config`. 
    - Please find compatible configurations in the following table:

        |Script           |Configurations|Multi-GPU|
        |-----------------|--------------|:-------:|
        |`train_gan.py`   |`GN-GAN_CIFAR10_CNN.txt`<br>`GN-GAN_CIFAR10_RES.txt`<br>`GN-GAN_CIFAR10_BIGGAN.txt`<br>`GN-GAN_STL10_CNN.txt`<br>`GN-GAN_STL10_RES.txt`<br>`GN-GAN-CR_CIFAR10_CNN.txt`<br>`GN-GAN-CR_CIFAR10_RES.txt`<br>`GN-GAN-CR_CIFAR10_BIGGAN.txt`<br>`GN-GAN-CR_STL10_CNN.txt`<br>`GN-GAN-CR_STL10_RES.txt`||
        |`train_ddp.py`|`GN-GAN_CELEBAHQ128_RES.txt`<br>`GN-GAN_CELEBAHQ256_RES.txt`<br>`GN-GAN_CHURCH256_RES.txt`|:heavy_check_mark:|

- Run the training script with the compatible configuration, e.g.,
    - `train.py` supports training gan on `CIFAR10` and `STL10`, e.g.,
        ```sh
        python train.py \
            --flagfile ./config/GN-GAN_CIFAR10_RES.txt \
            --logdir ./logs/GN-GAN_CIFAR10_RES
        ```
    - `train_ddp.py` is optimized for multi-gpu training, e.g.,
        ```
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ddp.py \
            --flagfile ./config/GN-GAN_CELEBAHQ256_RES.txt \
            --logdir ./logs/GN-GAN_CELEBAHQ256_RES_0
        ```

- Generate images from checkpoints, e.g.,
    ```
    python train.py \
        --flagfile ./logs/GN-GAN_CIFAR10_RES/flagfile.txt \
        --eval \
        --save path/to/generated/images
    ```
