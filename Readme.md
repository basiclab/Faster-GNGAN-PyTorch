# Gradient Normalization for Generative Adversarial Networks

The authors' official implementation of Gradient Normalized GAN (GN-GAN).

## Requirements
- CUDA 10.2
- Python 3.9.12
- Python packages
    ```sh
    # update `pip` for installing latest tensorboard.
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10
    ```
    python -m training.datasets --dataset cifar10 --out ./data/cifar10
    ```

- STL-10
    ```
    python -m training.datasets --dataset stl10 --out ./data/stl10
    ```

- CelebA-HQ

    We obtain celeba-hq from [this repository](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download) and save them in `lmdb`.
    
    ```
    python -m training.datasets --dataset path/to/celebahq --out ./data/celebahq
    ```

- LSUN Church
    ```
    python -m training.datasets --dataset path/to/church --out ./data/church
    ```

## Preprocessing Datasets for FID
- Download pre-calculated statistic for FID from [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing).

- The folder structure:
    ```
    ./stats
    ├── celebahq.all.256.npz
    ├── church.train.256.npz
    ├── cifar10.test.npz
    ├── cifar10.train.npz
    └── stl10.unlabeled.48.npz
    ```

**NOTE**

All the reported values in our paper are calculated by official implementation of Inception Score and FID.


## Training
- All the configurations can be found in `./configs`. 
- Train from scratch:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_train.py \
        --config ./config/GN_cifar10_resnet.txt \
        --logdir ./logs/GN_cifar10_resnet_0
    ```
- Multi-GPU training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_train.py \
        --config ./config/GN_cifar10_resnet.txt \
        --logdir ./logs/GN_cifar10_resnet_0
    ```
