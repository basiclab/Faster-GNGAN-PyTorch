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
python fid_score.py --path $1/generate --num_images 50000 --stats ./stats/stl10_train_unlabeled_48_AutoGAN.npz
python fid_score.py --path $1/generate --num_images 10000 --stats ./stats/stl10_train_unlabeled_48_AutoGAN.npz
deactivate
