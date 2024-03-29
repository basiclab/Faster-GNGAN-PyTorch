# On the Effectiveness of Gradient Normalized Generative Adversarial Networks

This is the official implementation of Faster Gradient Normalized GAN (Faster GN-GAN) by the authors.

## Requirements
- CUDA 11.3
- Python packages
    ```sh
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10 and STL-10

    We use the PyTorch built-in dataset for CIFAR-10 and STL-10.

- CelebA-HQ

    We obtain CelebA-HQ from [this repository](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download) and preprocess them into `lmdb` format using the following command:

    ```
    python -m training.datasets --dataset celebahq/images --out ./data/celebahq
    ```

- LSUN Church

    We obtain LSUN Church from official [website](https://www.yf.io/p/lsun).

### Folder Structure
```
./data
├── celebahq
│   ├── data.mdb
│   └── lock.mdb
├── cifar10 (created by pytorch)
├── lsun
│   └── church_outdoor_train_lmdb
│       ├── data.mdb
│       └── lock.mdb
└── stl10 (created by pytorch)
```

## Preprocessing Datasets for FID
- Download pre-calculated statistic from [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing) to calculating FID.

- The folder structure should be as follows:
    ```
    ./stats
    ├── celebahq.all.256.npz
    ├── church.train.256.npz
    ├── cifar10.test.npz
    ├── cifar10.train.npz
    └── stl10.unlabeled.48.npz
    ```

**NOTE**

All the values reported in our paper are calculated using the official implementation of Inception Score and FID.


## Training

All the configurations can be found in `./configs`.

- To train GN-GAN from scratch:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config ./config/GN_cifar10_resnet.json \
        --normalize_G training.gn.normalize_D \
        --logdir ./logs/GN_cifar10_resnet_0
    ```

- To train Faster GN-GAN from scratch:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config ./config/GN_cifar10_resnet.json \
        --normalize_G training.gn.normalize_G \
        --logdir ./logs/GN_cifar10_resnet_0
    ```

- To train Faster GN-GAN with rescaling from scratch:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config ./config/GN_cifar10_resnet.json \
        --normalize_G training.gn.normalize_G \
        --scale 0 \
        --logdir ./logs/GN_cifar10_resnet_0
    ```

- To train GN-GAN with multi-GPU:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
        ...
    ```
