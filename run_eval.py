import os
import tempfile

import click
import torch
import torchvision
from pytorch_gan_metrics import get_fid, get_fid_from_directory
from tqdm import tqdm

from training import misc


@click.command(cls=misc.CommandAwareConfig('config'))
@click.option('--config', default=None, type=str)
@click.option('--logdir', default='./logs/GN_cifar10_resnet')
@click.option('--resolution', default=32)
@click.option('--n_classes', default=1)
@click.option('--z_dim', default=128)
@click.option('--architecture_G', 'architecture_G', default='training.models.resnet.Generator')
@click.option('--bs_G', 'bs_G', default=128)
@click.option('--accumulation', default=1)
@click.option('--eval_size', default=50000)
@click.option('--fid_stats', default='./stats/cifar10.train.npz')
@click.option('--output', default=None, type=str)
@click.option('--ema/--no-ema', default=False)
@click.option('--seed', default=0)
def main(**kwargs):
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))
    if num_gpus == 1:
        subprocess_fn(0, num_gpus, kwargs)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            processes = []
            for rank in range(num_gpus):
                p = torch.multiprocessing.Process(
                    target=subprocess_fn, args=(rank, num_gpus, kwargs, temp_dir))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


def subprocess_fn(rank, num_gpus, kwargs, temp_dir=None):
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        init_method = f'file://{init_file}'
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus)
        print("Node %d is initialized" % rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    evaluating_loop(rank, num_gpus, **kwargs)


def evaluating_loop(
    rank: int,              # Rank of the current process in [0, num_gpus[.
    num_gpus: int,          # Number of GPUs participating in the training.
    logdir: str,            # Directory where to save the model and tf board.
    resolution: int,        # Resolution of the images.
    n_classes: int,         # Number of classes in the dataset.
    z_dim: int,             # Dimension of latent space.
    architecture_G: str,    # Generator class path.
    bs_G: int,              # Total batch size for one training iteration of G.
    eval_size: int,         # Number of images to evaluate.
    fid_stats: str,         # Path to the FID statistics.
    output: str,            # Path to output directory. None for no output.
    ema: bool,              # Whether to evaluate exponential moving averaged G.
    seed: int,              # Seed for random number generators.
    **dummy,
):
    assert bs_G % num_gpus == 0, "bs_G is not divisible by accumulation and num_gpus"
    bs_G = bs_G // num_gpus
    misc.set_seed(rank + seed)

    device = torch.device('cuda:%d' % rank)
    G = misc.construct_class(architecture_G, resolution, n_classes, z_dim).to(device)

    # Initialize models for multi-gpu training.
    is_ddp = num_gpus > 1
    if is_ddp:
        G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[rank], output_device=rank)

    # Load the model.
    if ema:
        G_state_dict = torch.load(
            os.path.join(logdir, 'best.ema.pt'), map_location='cpu')['G_ema']
    else:
        G_state_dict = torch.load(
            os.path.join(logdir, 'best.pt'), map_location='cpu')['G']
    G.load_state_dict(G_state_dict)

    # if `--output` is used, the generated images are saved to the output
    # directory on the fly instead of storing them in RAM. Which is useful when
    # generating high resolution images in a low memory system. The FID is
    # therefore calculated by `get_fid_from_directory`.
    if rank == 0:
        if output:
            os.makedirs(output, exist_ok=True)
            counter = 0
        else:
            imgs = []
    progress = tqdm(total=eval_size, ncols=0, desc="Generating", leave=False, disable=rank != 0)
    # Generate images.
    for _ in range(0, eval_size, bs_G * num_gpus):
        z = torch.randn(bs_G, z_dim, device=device)
        y = torch.randint(n_classes, (bs_G,), device=device)
        with torch.no_grad():
            batch_imgs = G(z, y)
        if num_gpus > 1:
            buf = [torch.empty_like(batch_imgs) for _ in range(num_gpus)]
            torch.distributed.all_gather(buf, batch_imgs)
            batch_imgs = torch.cat(buf, dim=0).cpu()
        if rank == 0:
            if output:
                for img in batch_imgs:
                    if counter < eval_size:
                        path = os.path.join(output, f'{counter}.png')
                        torchvision.utils.save_image(img, path)
                        counter += 1
            else:
                imgs.append(batch_imgs)
            progress.update(bs_G * num_gpus)
    progress.close()
    del G

    if rank != 0:
        return

    if output:
        # Calculate FID.
        FID = get_fid_from_directory(output, fid_stats, verbose=True)
    else:
        imgs = torch.cat(imgs, dim=0)[:eval_size]
        assert len(imgs) == eval_size
        # Calculate FID.
        FID = get_fid(imgs, fid_stats, verbose=True)

    print(f"FID: {FID:.3f}")


if __name__ == '__main__':
    main()
