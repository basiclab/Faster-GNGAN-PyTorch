#! /bin/bash
#[logdir]

cd /home/$USER/workspace/pytorch-gan-collections/
source venv/bin/activate
python gans/train_gan.py --flagfile $1/flagfile.txt --logdir $1 --generate --num_images 50000
deactivate

cd /home/$USER/workspace/improved-gan
source venv/bin/activate
python inception_score.py --path $1/generate --num_images 50000
deactivate

cd /home/$USER/workspace/TTUR
source venv/bin/activate
python fid_score.py --path $1/generate --num_images 50000 --stats ./stats/cifar10_train.npz
python fid_score.py --path $1/generate --num_images 10000 --stats ./stats/cifar10_test.npz
deactivate
