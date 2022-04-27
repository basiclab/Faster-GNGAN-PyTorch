import os
import tempfile

import click
import torch

from training.training_loop import training_loop
from training import misc


def subprocess_fn(rank, num_gpus, kwargs, temp_dir=None):
    if num_gpus > 1:
        init_file = os.path.abspath(
            os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(
            backend='nccl', init_method=init_method,
            rank=rank, world_size=num_gpus)
        print("Node %d is initialized" % rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    training_loop(rank, num_gpus, kwargs=kwargs, **kwargs)


@click.command(cls=misc.CommandAwareConfig('config'))
@click.option('--config', default=None, type=str)
@click.option('--resume/--no-resume', default=False)
@click.option('--logdir', default='./logs/GN_cifar10_resnet')
@click.option('--data_path', default='./data/cifar10')
@click.option('--hflip/--no-hflip', default=True)
@click.option('--resolution', default=32)
@click.option('--n_classes', default=1)
@click.option('--z_dim', default=128)
@click.option('--architecture_d', 'architecture_D', default='training.models.resnet.Discriminator')
@click.option('--architecture_g', 'architecture_G', default='training.models.resnet.Generator')
@click.option('--loss', default='training.losses.HingeLoss')
@click.option('--steps', default=200000)
@click.option('--step_d', 'step_D', default=5)
@click.option('--bs_d', 'bs_D', default=64)
@click.option('--bs_g', 'bs_G', default=128)
@click.option('--lr_d', 'lr_D', default=0.0004)
@click.option('--lr_g', 'lr_G', default=0.0002)
@click.option('--lr_decay/--no-lr_decay', default=True)
@click.option('--accumulation', default=1)
@click.option('--beta0', default=0.0)
@click.option('--beta1', default=0.9)
@click.option('--cr_gamma', default=0.0)
@click.option('--gp_gamma', default=0.0)
@click.option('--use_gn/--no-use_gn', default=True)
@click.option('--rescale_alpha', default=None, type=float)
@click.option('--ema_decay', default=0.9999)
@click.option('--ema_start', default=0)
@click.option('--sample_step', default=500)
@click.option('--sample_size', default=64)
@click.option('--eval_step', default=5000)
@click.option('--eval_size', default=50000)
@click.option('--fid_stats', default='./stats/cifar10.train.npz')
@click.option('--save_step', default=20000)
@click.option('--seed', default=0)
def main(**kwargs):
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))
    if num_gpus == 1:
        subprocess_fn(0, num_gpus, kwargs)
    else:
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(
                target=subprocess_fn, args=(num_gpus, kwargs, temp_dir), nprocs=num_gpus)


if __name__ == '__main__':
    main()
