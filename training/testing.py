import os

import torch
import torchvision
from pytorch_gan_metrics import (
    get_inception_score_and_fid, get_inception_score_and_fid_from_directory)
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from training import misc
from training import dist


def testing(
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
    assert bs_G % dist.num_gpus() == 0, "bs_G is not divisible by accumulation and num_gpus"
    bs_G = bs_G // dist.num_gpus()
    misc.set_seed(dist.rank() + seed)

    device = dist.device()
    G = misc.dynamic_import(architecture_G)(resolution, n_classes, z_dim).to(device)

    # Initialize models for multi-gpu inferencing.
    if dist.is_initialized():
        G = SyncBatchNorm.convert_sync_batchnorm(G)
        G = DistributedDataParallel(G, device_ids=[device])
    G.requires_grad_(False)

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
    if dist.is_main():
        if output:
            os.makedirs(output, exist_ok=True)
            counter = 0
        else:
            imgs = []
    progress = tqdm(
        total=eval_size,
        ncols=0,
        desc="Generating",
        leave=False,
        disable=not dist.is_main())
    # Generate images.
    for _ in range(0, eval_size, bs_G * dist.num_gpus()):
        z = torch.randn(bs_G, z_dim, device=device)
        y = torch.randint(n_classes, (bs_G,), device=device)
        with torch.no_grad():
            batch_imgs = (G(z, y) + 1) / 2
            batch_imgs = dist.gather(batch_imgs)
        if dist.is_main():
            if output:
                for img in batch_imgs:
                    if counter < eval_size:
                        path = os.path.join(output, f'{counter}.png')
                        torchvision.utils.save_image(img, path)
                        counter += 1
            else:
                imgs.append(batch_imgs.cpu())
            progress.update(bs_G * dist.num_gpus())
    progress.close()
    del G

    if not dist.is_main():
        return

    if output:
        # Calculate FID.
        (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
            output, fid_stats, verbose=True)
    else:
        imgs = torch.cat(imgs, dim=0)[:eval_size]
        assert len(imgs) == eval_size
        # Calculate FID.
        (IS, IS_std), FID = get_inception_score_and_fid(
            imgs, fid_stats, verbose=True)

    print(f"IS    : {IS:.3f}")
    print(f"IS_std: {IS_std:.3f}")
    print(f"FID   : {FID:.3f}")
