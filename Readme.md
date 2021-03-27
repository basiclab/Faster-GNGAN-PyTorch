# Gradient Normalization for Generative Adversarial Networks

The author's official implementation of GN-GAN.

## System Requirements
- CUDA 10.2
- CUDNN 7.5.x
- Python 3.6.9
- Python packages
    ```
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10

    Pytorch build-in CIFAR-10 will be automatically downloaded.

- STL-10

    Pytorch build-in STL-10 will be automatically downloaded.

- CelebA-HQ 256x256

    Unzip dataset and create folder follow the structer as below.
    ```
    ./data/celebhq/dummy
    ├── 00001.jpg
    ├── 00002.jpg
    ├── ...
    └── 30000.jpg
    ```

- [LSUN Church Outdoor](https://www.yf.io/p/lsun) 256x256

    Unzip dataset and create folder follow the structer 
    ```
    ./data/lsun/church_outdoor_train_lmdb/
    ├── data.mdb
    └── lock.mdb
    ```

- Preprocess CelebA-HQ and LSUN Church Outdoor for speeding up IO
    ```
    python source/dataset.py
    ```

## Statistic for FID
Please follow the `metrics/Readme.md` to create following 5 statistics:
- cifar10.train.npz - Training set of CIFAR10
- cifar10.train.npz - Testing set of CIFAR10
- stl10.unlabeled.48.npz - Unlabeled set of STL10 in resolution 48x48
- church_outdoor.train.256.npz - Center Cropped training set of LSUN Church Outdoor
- celebhq.all.256.npz - Full dataset of CelebA-HQ 256x256

Then, create `stats` and put all above 5 statistics to `./stats`.


## How to Training
- There are 3 training scripts , i.e.,
    - `train_gan.py`
    - `train_gan_large_dist.py`
    - `train_cgan.py`
    
    Training configurations are located in `./config`. Moreover, the following table is the compatible configuration list.

    |Script           |Configurations|Multi-GPU|
    |-----------------|--------------|:-------:|
    |`train_gan.py`   |`GN-GAN_CIFAR10_CNN.txt`,<br>`GN-GAN_CIFAR10_RES.txt`,<br>`GN-GAN_STL10_CNN.txt`,<br>`GN-GAN_STL10_RES.txt`,<br>`GN-GAN-CR_CIFAR10_CNN.txt`,<br>`GN-GAN-CR_CIFAR10_RES.txt`,<br>`GN-GAN-CR_STL10_CNN.txt`,<br>`GN-GAN-CR_STL10_RES.txt`||
    |`train_gan_large_dist.py`|`GN-GAN_CELEBHQ256_RES.txt`,<br>`GN-GAN_CHURCH256_RES.txt`|:heavy_check_mark:|
    |`train_cgan.py`  |`GN-cGAN_CIFAR10_BIGGAN.txt`,<br>`GN-cGAN-CR_CIFAR10_BIGGAN.txt`||

- Run the training script with the compatible configuration, e.g.,
    ```
    python train_gan.py \
        --flagfile ./config/GN-GAN_CIFAR10_RES.txt \
        --logdir ./logs/GN-GAN_CIFAR10_RES
    ```
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_gan_large_dist.py \
        --flagfile ./config/GN-GAN_CELEBHQ256_RES.txt \
        --logdir ./logs/GN-GAN_CELEBHQ256_RES_0
    ```

- Generate images from pretrained model, e.g.,
    ```
    python gans/train_gan.py \
        --flagfile ./logs/GN-GAN_CIFAR10_RES/flagfile.txt \
        --generate \
        --output ./generated_images
    ```

    The generated samples are saved into `./generated_images`
