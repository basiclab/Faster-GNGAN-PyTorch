import copy
import json
import os
from random import sample, shuffle

import torch
import torchvision
from pytorch_gan_metrics import get_fid
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm

from training import datasets
from training import misc
from training import gn


def infiniteloop(dataloader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for x in dataloader:
            yield x


def fid(
    device,                 # torch device.
    rank,                   # Rank of the current process in [0, num_gpus[.
    num_gpus,               # Number of GPUs participating in the training.
    G,                      # Generator.
    bs_G: int,              # Batch size for G.
    z_dim: int,             # Dimension of latent space.
    n_classes: int,         # Number of classes in the dataset.
    eval_size: int,         # Number of images to evaluate.
    fid_stats: str,         # Path to the FID statistics.
    **kwargs
):
    imgs = []
    progress = tqdm(
        total=eval_size,
        ncols=0,
        desc="Generating",
        leave=False,
        disable=rank != 0)
    for _ in range(0, eval_size, bs_G * num_gpus):
        z = torch.randn(bs_G, z_dim, device=device)
        y = torch.randint(n_classes, (bs_G,), device=device)
        with torch.no_grad():
            batch_imgs = (G(z, y) + 1) / 2
        if num_gpus > 1:
            buf = [torch.empty_like(batch_imgs) for _ in range(num_gpus)]
            torch.distributed.all_gather(buf, batch_imgs)
            batch_imgs = torch.cat(buf, dim=0).cpu()
        imgs.append(batch_imgs)
        progress.update(bs_G * num_gpus)
    progress.close()
    if rank == 0:
        imgs = torch.cat(imgs, dim=0)[:eval_size]
        assert len(imgs) == eval_size
        FID = get_fid(imgs, fid_stats, verbose=True)
    else:
        FID = None
    if num_gpus > 1:
        torch.distributed.barrier()
    del imgs
    return FID


def train_D(
    device,                 # torch device.
    loader,                 # Iterator of DataLoader.
    loss_meter,             # Meter for recording loss value.
    D,                      # Discriminator.
    G,                      # Generator.
    loss_fn,                # Loss function.
    gain,                   # Loss gain. Used for gradient accumulation.
    bs_D: int,              # Batch size for D.
    n_classes: int,         # Number of classes in the dataset.
    z_dim: int,             # Dimension of latent space.
    cr_gamma: float,        # Consistency regularization gamma.
    gp0_gamma: float,       # 0 Gradient penalty gamma.
    gp1_gamma: float,       # 1 Gradient penalty gamma.
    gps_gamma: float,       # sign Gradient penalty gamma.
    use_gn: bool,           # Whether to use gradient normalization.
    use_fn: bool,           # Whether to use functional normalized GN.
    use_gn_D: bool,         # Whether to use gradient normalization in D.
    c: float,               #
    **kwargs,
):
    images_real, classes_real, images_aug = next(loader)
    images_real, classes_real = images_real.to(device), classes_real.to(device)
    z = torch.randn(bs_D, z_dim, device=device)
    classes_fake = torch.randint(n_classes, (bs_D,), device=device)
    with torch.no_grad():
        images_fake = G(z, classes_fake)
    x = torch.cat([images_real, images_fake], dim=0)
    y = torch.cat([classes_real, classes_fake], dim=0)
    if gp0_gamma > 0 or gp1_gamma > 0 or gps_gamma > 0:
        x.requires_grad_(True)
    if use_gn or use_gn_D:
        scores = gn.normalize_D(D, x, loss_fn, use_fn, c, y=y)
    else:
        scores = D(x, y=y)
    scores_real, scores_fake = torch.split(scores, bs_D)
    loss_fake, loss_real = loss_fn(scores_fake, scores_real)
    loss_D = loss_fake + loss_real
    loss_meter.append('loss/D', (loss_fake + loss_real).detach().cpu())
    loss_meter.append('loss/D/real', loss_real.detach().cpu())
    loss_meter.append('loss/D/fake', loss_fake.detach().cpu())

    # Consistency Regularization.
    if cr_gamma > 0:
        images_aug = images_aug.to(device)
        if use_gn or use_gn_D:
            scores_aug = gn.normalize_D(D, images_aug, loss_fn, use_fn, c, y=classes_real)
        else:
            scores_aug = D(images_aug, y=classes_real)
        loss_cr = (scores_aug - scores_real).square().mean()
        loss_D += loss_cr.mul(cr_gamma)
        loss_meter.append('loss/D/cr', loss_cr.detach().cpu())

    # Gradient Penalty.
    if gp0_gamma > 0 or gp1_gamma > 0 or gps_gamma > 0:
        grad = torch.autograd.grad(
            outputs=scores.sum(), inputs=x, create_graph=True)[0]
        grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
        if gp0_gamma > 0:
            loss_gp = grad_norm
            loss_D += loss_gp.mul(gp0_gamma).mean()
            loss_meter.append('loss/D/gp0', loss_gp.detach().mean().cpu())
        if gp1_gamma > 0:
            loss_gp = (grad_norm - 1).square()
            loss_D += loss_gp.mul(gp1_gamma).mean()
            loss_meter.append('loss/D/gp1', loss_gp.detach().mean().cpu())
        if gps_gamma > 0:
            f_sign = torch.cat([torch.ones(bs_D, 1), -torch.ones(bs_D, 1)]).to(device)
            g_sign = scores.detach().sign()
            loss_gp = grad_norm
            loss_D += loss_gp.mul(f_sign).mul(g_sign).mul(gps_gamma).mean()
            loss_meter.append('loss/D/gps', loss_gp.detach().mean().cpu())

    x.retain_grad()
    # Backward.
    loss_D.mul(gain).backward()

    with torch.no_grad():
        grad_norm_lower_bound = (1 + x.flatten(start_dim=1).norm(dim=1)).square().reciprocal().mean()
        grad_norm = (x.grad.flatten(start_dim=1).norm(dim=1) * x.shape[0] / 2).mean()
        margin = grad_norm - grad_norm_lower_bound
        loss_meter.append('misc/grad_norm_lower_bound', grad_norm_lower_bound.cpu())
        loss_meter.append('misc/grad_norm', grad_norm.cpu())
        loss_meter.append('misc/margin', margin.cpu())


def train_G(
    device,                 # torch device.
    loss_meter,             # Meter for recording loss value.
    D,                      # Discriminator.
    G,                      # Generator.
    loss_fn,                # Loss function.
    gain,                   # Loss gain. Used for gradient accumulation.
    bs_G: int,              # Batch size for G.
    n_classes: int,         # Number of classes in the dataset.
    z_dim: int,             # Dimension of latent space.
    use_gn: bool,           # Whether to use gradient normalization.
    use_fn: bool,           # Whether to use functional normalized GN.
    use_gn_G: bool,         # Whether to use gradient normalization in G.
    c: float,               #
    **kwargs,
):
    z = torch.randn(bs_G, z_dim, device=device)
    y = torch.randint(n_classes, (bs_G,), device=device)
    fake = G(z, y)
    if use_gn or use_gn_G:
        scores = gn.normalize_G(D, fake, loss_fn, use_fn, c, y=y)
        loss_G = scores.mean()
    else:
        scores = D(fake, y=y)
        loss_G = loss_fn(scores)
    loss_G.mul(gain).backward()

    loss_meter.append('loss/G', loss_G.detach().cpu())


def training_loop(
    rank: int,              # Rank of the current process in [0, num_gpus[.
    num_gpus: int,          # Number of GPUs participating in the training.
    resume: bool,           # Whether to resume training from a logdir.
    logdir: str,            # Directory where to save the model and tf board.
    data_path: str,         # Path to the dataset.
    hflip: bool,            # Horizontal flip augmentation.
    resolution: int,        # Resolution of the images.
    n_classes: int,         # Number of classes in the dataset.
    z_dim: int,             # Dimension of latent space.
    architecture_D: str,    # Discriminator class path.
    architecture_G: str,    # Generator class path.
    loss_D: str,            # loss function class path for D.
    loss_G: str,            # loss function class path for G.
    steps: int,             # Total iteration of the training.
    step_D: int,            # The number of iteration of the D per iteration of the G.
    bs_D: int,              # Total batch size for one training iteration of D.
    bs_G: int,              # Total batch size for one training iteration of G.
    lr_D: float,            # Learning rate of the D.
    lr_G: float,            # Learning rate of the G.
    lr_decay: bool,         # Whether to linearly decay the learning rate.
    accumulation: int,      # Number of gradient accumulation.
    beta0: float,           # Beta0 of the Adam optimizer.
    beta1: float,           # Beta1 of the Adam optimizer.
    cr_gamma: float,        # Consistency regularization gamma.
    gp0_gamma: float,       # 0 Gradient penalty gamma.
    gp1_gamma: float,       # 1 Gradient penalty gamma.
    gps_gamma: float,       # sign Gradient penalty gamma.
    use_gn: bool,           # Whether to use gradient normalization.
    use_fn: bool,           # Whether to use functional normalized GN.
    use_gn_D: bool,         # Whether to use gradient normalization in D.
    use_gn_G: bool,         # Whether to use gradient normalization in G,
    rescale_alpha: float,   # Alpha parameter of the rescaling.
    c,
    ema_decay: float,       # Decay rate of the exponential moving average.
    ema_start: int,         # Start iteration of the exponential moving average.
    sample_step: int,       # Sample from fixed z every sample_step iterations.
    sample_size: int,       # Number of samples to generate.
    eval_step: int,         # Evaluate the model every eval_step iterations.
    eval_size: int,         # Number of images to evaluate.
    fid_stats: str,         # Path to the FID statistics.
    save_step: int,         # Save the model every save_step iterations.
    seed: int,              # Seed for random number generators.
    kwargs: dict,           # All arguments for dumping to the config file.
    **dummy,
):
    assert bs_D % (accumulation * num_gpus) == 0, "bs_D is not divisible by accumulation and num_gpus"
    assert bs_G % (accumulation * num_gpus) == 0, "bs_G is not divisible by accumulation and num_gpus"
    bs_D = bs_D // (accumulation * num_gpus)
    bs_G = bs_G // (accumulation * num_gpus)
    misc.set_seed(rank + seed)

    device = torch.device('cuda:%d' % rank)
    dataset = datasets.Dataset(data_path, hflip, resolution, cr_gamma > 0)
    sampler = DistributedSampler(dataset, num_gpus, rank, seed=seed, drop_last=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs_D,
        sampler=sampler,
        num_workers=min(torch.get_num_threads(), 16),
        drop_last=True)
    loader = infiniteloop(loader, sampler)

    # Construct Models.
    D = misc.construct_class(architecture_D, resolution, n_classes).to(device)
    G = misc.construct_class(architecture_G, resolution, n_classes, z_dim).to(device)
    G_ema = copy.deepcopy(G)

    # Initialize models for multi-gpu training.
    is_ddp = num_gpus > 1
    if is_ddp:
        D = torch.nn.parallel.DistributedDataParallel(
            D, device_ids=[rank], output_device=rank)
        G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
        G = torch.nn.parallel.DistributedDataParallel(
            G, device_ids=[rank], output_device=rank)
        G_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_ema)
        G_ema = torch.nn.parallel.DistributedDataParallel(
            G_ema, device_ids=[rank], output_device=rank)

    # Initialize Optimizer.
    D_opt = torch.optim.Adam(D.parameters(), lr=lr_D, betas=[beta0, beta1])
    G_opt = torch.optim.Adam(G.parameters(), lr=lr_G, betas=[beta0, beta1])

    # Setup learning rate linearly decay scheduler.
    def decay_rate(step):
        if lr_decay:
            return 1 - step / steps
        else:
            return 1.0
    D_lrsched = torch.optim.lr_scheduler.LambdaLR(D_opt, lr_lambda=decay_rate)
    G_lrsched = torch.optim.lr_scheduler.LambdaLR(G_opt, lr_lambda=decay_rate)

    # Loss function for real and fake images.
    loss_fn_D = misc.construct_class(loss_D)
    loss_fn_G = misc.construct_class(loss_G)

    # tf board writer.
    if rank == 0:
        writer = SummaryWriter(logdir)
        if is_ddp:
            collector = misc.collect_forward_backward_norm(D.module)
        else:
            collector = misc.collect_forward_backward_norm(D)

    if not resume:
        # Sample fixed random noises and classes.
        fixed_z = torch.randn(sample_size, z_dim, device=device)
        fixed_z = torch.split(fixed_z, sample_size // num_gpus, dim=0)
        fixed_y = torch.randint(n_classes, (sample_size,), device=device)
        fixed_y = torch.split(fixed_y, sample_size // num_gpus, dim=0)
        # Initialize iteration and best results.
        start_step = 0
        best = {
            'FID/best': float('inf'),
            'FID/ema/best': float('inf'),
        }
        if rank == 0:
            # Save arguments fo config.json
            with open(os.path.join(logdir, "config.json"), 'w') as f:
                json.dump(kwargs, f, indent=2, sort_keys=True)
            samples = [dataset[i][0] for i in range(sample_size)]
            writer.add_image('real', torchvision.utils.make_grid(samples))
            writer.flush()
    else:
        ckpt = torch.load(os.path.join(logdir, 'model.pt'), map_location='cpu')
        D.load_state_dict(ckpt['D'])
        G.load_state_dict(ckpt['G'])
        G_ema.load_state_dict(ckpt['G_ema'])
        D_opt.load_state_dict(ckpt['D_opt'])
        G_opt.load_state_dict(ckpt['G_opt'])
        D_lrsched.load_state_dict(ckpt['D_lrsched'])
        G_lrsched.load_state_dict(ckpt['G_lrsched'])
        fixed_z = ckpt['fixed_z'].to(device)
        fixed_z = torch.split(fixed_z, sample_size // num_gpus, dim=0)
        fixed_y = ckpt['fixed_y'].to(device)
        fixed_y = torch.split(fixed_y, sample_size // num_gpus, dim=0)
        start_step = ckpt['step']
        best = ckpt['best']
        del ckpt

    kwargs['bs_D'] //= (accumulation * num_gpus)
    kwargs['bs_G'] //= (accumulation * num_gpus)

    progress = trange(
        start_step + 1,         # Initial step value.
        steps + 1,              # The value is from 1 to steps (include).
        initial=start_step,     # Initial progress value.
        total=steps,            # The progress size.
        ncols=0,                # Disable bar, only show steps and percentage.
        desc='Training',
        disable=(rank != 0))

    for step in progress:
        meter = misc.Meter()

        # Update D.
        D.requires_grad_(True)
        for _ in range(step_D):
            if rescale_alpha is not None:
                if is_ddp:
                    D.module.rescale(alpha=rescale_alpha)
                else:
                    D.rescale(alpha=rescale_alpha)
            for i in range(accumulation):
                with misc.ddp_sync(D, sync=(i == accumulation - 1)):
                    train_D(device, loader, meter, D, G, loss_fn_D,
                            gain=1 / accumulation, **kwargs)
            D_opt.step()
            D_opt.zero_grad(set_to_none=True)
        D_lrsched.step()
        D.requires_grad_(False)

        if rank == 0:
            # Record the last forward and backward pass of D.
            for tag, value in collector.norms():
                writer.add_scalar(tag, value, step)

        # Update G.
        G.requires_grad_(True)
        for i in range(accumulation):
            with misc.ddp_sync(G, sync=(i == accumulation - 1)):
                train_G(device, meter, D, G, loss_fn_G,
                        gain=1 / accumulation, **kwargs)
        G_opt.step()
        G_opt.zero_grad(set_to_none=True)
        G_lrsched.step()
        G.requires_grad_(False)

        # Update G_ema.
        ema_beta = ema_decay if step > ema_start else 0
        G_dict = G.state_dict()
        G_ema_dict = G_ema.state_dict()
        for name in G_dict.keys():
            G_ema_dict[name].data.copy_(
                G_ema_dict[name].data * ema_beta + G_dict[name].data * (1 - ema_beta))

        if rank == 0:
            # Update tf board
            losses = meter.todict()
            for tag, value in losses.items():
                writer.add_scalar(tag, value, step)

            # Update progress bar
            progress.set_postfix_str(", ".join([
                f"D_fake: {losses['loss/D/fake']:.3f}",
                f"D_real: {losses['loss/D/real']:.3f}",
                f"G: {losses['loss/G']:.3f}",
            ]))

        # Generate images from fixed z every sample_step steps.
        if step == 1 or step % sample_step == 0:
            with torch.no_grad():
                imgs = (G(fixed_z[rank], fixed_y[rank]) + 1) / 2
                imgs_ema = (G_ema(fixed_z[rank], fixed_y[rank]) + 1) / 2
            if is_ddp:
                buf = [torch.empty_like(imgs) for _ in range(num_gpus)]
                buf_ema = [torch.empty_like(imgs_ema) for _ in range(num_gpus)]
                torch.distributed.all_gather(buf, imgs)
                torch.distributed.all_gather(buf_ema, imgs_ema)
                imgs = torch.cat(buf, dim=0).cpu()
                imgs_ema = torch.cat(buf_ema, dim=0).cpu()
            if rank == 0:
                writer.add_image('fake', torchvision.utils.make_grid(imgs), step)
                writer.add_image('fake/ema', torchvision.utils.make_grid(imgs_ema), step)

        # Calculate FID every eval_step steps.
        if step == 1 or step % eval_step == 0:
            FID = fid(device, rank, num_gpus, G, **kwargs)
            FID_ema = fid(device, rank, num_gpus, G_ema, **kwargs)
            if rank == 0:
                if FID < best['FID/best']:
                    best['FID/best'] = FID
                    save_best_model = True
                else:
                    save_best_model = False
                if FID_ema < best['FID/ema/best']:
                    best['FID/ema/best'] = FID_ema
                    save_best_ema_model = True
                else:
                    save_best_ema_model = False
                ckpt = {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'G_ema': G_ema.state_dict(),
                    'G_opt': G_opt.state_dict(),
                    'D_opt': D_opt.state_dict(),
                    'G_lrsched': G_lrsched.state_dict(),
                    'D_lrsched': D_lrsched.state_dict(),
                    'fixed_z': torch.cat(fixed_z, dim=0),
                    'fixed_y': torch.cat(fixed_y, dim=0),
                    'best': best,
                    'step': step,
                }
                torch.save(ckpt, os.path.join(logdir, 'model.pt'))
                if save_best_model:
                    torch.save(ckpt, os.path.join(logdir, 'best.pt'))
                if save_best_ema_model:
                    torch.save(ckpt, os.path.join(logdir, 'best.ema.pt'))
                if step == 1 or step % save_step == 0:
                    torch.save(ckpt, os.path.join(logdir, f'{step:06d}.pt'))
                metrics = {
                    'step': step,
                    'FID': FID,
                    'FID/ema': FID_ema,
                    **best,
                }
                for name, value in metrics.items():
                    if name != 'step':
                        writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(logdir, 'eval.txt'), 'a') as f:
                    f.write(json.dumps(metrics) + "\n")
                progress.write(", ".join([
                    f"{step:6d}/{steps:6d}",
                    f"FID: {FID:.3f}",
                    f"FID/ema: {FID_ema:.3f}",
                ]))

    if rank == 0:
        progress.close()
        writer.close()
