import os
import tempfile

import click
import torch

from training import training_loop
from training import testing
from training import misc


@click.command(cls=misc.CommandAwareConfig)
@click.option('--config', default=None, type=str)
@click.option('--resume/--no-resume', default=False)
@click.option('--logdir', default='./logs/GN_cifar10_resnet')
@click.option('--data_path', default='./data/cifar10')
@click.option('--hflip/--no-hflip', default=True)
@click.option('--resolution', default=32)
@click.option('--n_classes', default=1)
@click.option('--z_dim', default=128)
@click.option('--architecture_D', 'architecture_D', default='training.models.resnet.Discriminator')
@click.option('--architecture_G', 'architecture_G', default='training.models.resnet.Generator')
@click.option('--loss_D', 'loss_D', default='training.losses.HingeLoss')
@click.option('--loss_G', 'loss_G', default='training.losses.HingeLoss')
@click.option('--normalize_D', 'normalize_D', default='training.gn.normalize_D')
@click.option('--normalize_G', 'normalize_G', default='training.gn.normalize_G')
@click.option('--steps', default=200000)
@click.option('--step_D', 'step_D', default=5)
@click.option('--bs_D', 'bs_D', default=64)
@click.option('--bs_G', 'bs_G', default=128)
@click.option('--lr_D', 'lr_D', default=0.0004)
@click.option('--lr_G', 'lr_G', default=0.0002)
@click.option('--lr_decay/--no-lr_decay', default=True)
@click.option('--accumulation', default=1)
@click.option('--beta0', default=0.0)
@click.option('--beta1', default=0.9)
@click.option('--cr_lambda', default=0.0)
@click.option('--gp0_lambda', default=0.0)
@click.option('--gp1_lambda', default=0.0)
@click.option('--gn_impl', type=click.Choice(['norm_G', 'norm_D']), default='norm_G')
@click.option('--scale', default=None, type=float)
@click.option('--ema_decay', default=0.9999)
@click.option('--ema_start', default=0)
@click.option('--sample_step', default=500)
@click.option('--sample_size', default=64)
@click.option('--eval_step', default=5000)
@click.option('--eval_size', default=50000)
@click.option('--fid_stats', default='./stats/cifar10.train.npz')
@click.option('--save_step', default=20000)
@click.option('--seed', default=0)
# evaluation
@click.option('--test_only/--no-test_only', default=False)
@click.option('--output', type=str, default=None)
@click.option('--ema/--no-ema', default=False)
def main(**kwargs):
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))
    if num_gpus > 1:
        with tempfile.TemporaryDirectory() as temp_dir:
            processes = []
            for rank in range(num_gpus):
                p = torch.multiprocessing.Process(
                    target=subprocess_fn,
                    args=(rank, num_gpus, temp_dir, kwargs))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    else:
        if kwargs['test_only']:
            testing.testing(**kwargs, kwargs=kwargs)
        else:
            training_loop.training_loop(**kwargs, kwargs=kwargs)


def subprocess_fn(rank, num_gpus, temp_dir, kwargs):
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    init_method = f'file://{init_file}'
    torch.distributed.init_process_group('nccl', init_method, rank=rank, world_size=num_gpus)
    print("Node %d is initialized" % rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    if kwargs['test_only']:
        testing.testing(**kwargs, kwargs=kwargs)
    else:
        training_loop.training_loop(**kwargs, kwargs=kwargs)


if __name__ == '__main__':
    main()
