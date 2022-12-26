import copy
import json
import os

import torch
import torchvision
from pytorch_gan_metrics import get_fid
from tensorboardX import SummaryWriter
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm

from training import datasets
from training import misc
from training import gn
from training import dist


def infiniteloop(dataloader, sampler):
    epoch = 0
    while True:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for x in dataloader:
            yield x
        epoch += 1


def fid(
    G,                      # Generator.
    bs_G: int,              # Batch size for G.
    z_dim: int,             # Dimension of latent space.
    n_classes: int,         # Number of classes in the dataset.
    eval_size: int,         # Number of images to evaluate.
    fid_stats: str,         # Path to the FID statistics.
    **kwargs
):
    device = dist.device()
    imgs = []
    progress = tqdm(
        total=eval_size,
        ncols=0,
        desc="Generating",
        leave=False,
        disable=not dist.is_main())
    for _ in range(0, eval_size, bs_G * dist.num_gpus()):
        z = torch.randn(bs_G, z_dim, device=device)
        y = torch.randint(n_classes, (bs_G,), device=device)
        with torch.no_grad():
            batch_imgs = (G(z, y) + 1) / 2
            batch_imgs = dist.gather(batch_imgs)
        if dist.is_main():
            imgs.append(batch_imgs.cpu())
            progress.update(bs_G * dist.num_gpus())
    progress.close()
    if dist.is_main():
        imgs = torch.cat(imgs, dim=0)[:eval_size]
        assert len(imgs) == eval_size
        FID = get_fid(imgs, fid_stats, verbose=True)
    else:
        FID = None
    dist.barrier()
    del imgs
    return FID


def train_D(
    loader,                     # Iterator of DataLoader.
    meter: misc.Meter,          # Meter for recording loss value.
    D: torch.nn.Module,         # Discriminator.
    G: torch.nn.Module,         # Generator.
    loss_fn: callable,          # Loss function.
    normalize_fn: callable,     # Normalization function for D.
    bs_D: int,                  # Batch size for D.
    n_classes: int,             # Number of classes in the dataset.
    z_dim: int,                 # Dimension of latent space.
    cr_lambda: float,           # Consistency regularization gamma.
    gp0_lambda: float,            # 0 Gradient penalty gamma.
    gp1_lambda: float,           # 1 Gradient penalty gamma.
    **kwargs,
):
    device = dist.device()
    collector = misc.NablaHatFxCollector(D)

    images_real, classes_real, images_aug = next(loader)
    images_real, classes_real = images_real.to(device), classes_real.to(device)
    z = torch.randn(bs_D, z_dim, device=device)
    classes_fake = torch.randint(n_classes, (bs_D,), device=device)
    with torch.no_grad():
        images_fake = G(z, classes_fake)
    x = torch.cat([images_real, images_fake], dim=0)
    y = torch.cat([classes_real, classes_fake], dim=0)
    scores, (loss_real, loss_fake), norm_nabla_fx = normalize_fn(D, x, loss_fn, y=y)
    loss_D = loss_real.mean() + loss_fake.mean()

    # 1 Gradient Penalty
    if gp1_lambda > 0:
        t = torch.rand(bs_D, 1, 1, 1, device=device)
        inter_x = t * images_real + (1 - t) * images_fake
        inter_x.requires_grad_(True)
        inter_y, _, _ = normalize_fn(D, inter_x, loss_fn, y=classes_real)
        gp1 = torch.autograd.grad(
            outputs=inter_y.sum(), inputs=inter_x, create_graph=True)[0]
        gp1_norm = gp1.flatten(start_dim=1).norm(dim=1)
        loss_gp = (gp1_norm - 1).square().mean()
        loss_D += loss_gp.mul(gp1_lambda)
        meter.append('loss/D/gp', loss_gp.detach().cpu())

    # 0 Gradient Penalty
    if gp0_lambda > 0:
        loss_gp = norm_nabla_fx.square().mean()
        loss_D += loss_gp.mul(gp0_lambda)
        meter.append('loss/D/gp', loss_gp.detach().cpu())

    # Consistency Regularization.
    if cr_lambda > 0:
        scores_real, _ = scores.chunk(2, dim=0)
        images_aug = images_aug.to(device)
        scores_aug, _, _ = normalize_fn(D, images_aug, loss_fn, y=classes_real)
        loss_cr = (scores_aug - scores_real).square().mean()
        loss_D += loss_cr.mul(cr_lambda)
        meter.append('loss/D/cr', loss_cr.detach().cpu())

    # Backward.
    loss_D.backward()
    # Remove hooks to avoid memory leak.
    collector.remove()

    meter.append('loss/D', (loss_real + loss_fake).mean().detach().cpu())
    meter.append('loss/D/real', loss_real.mean().detach().cpu())
    meter.append('loss/D/fake', loss_fake.mean().detach().cpu())
    with torch.no_grad():
        # Record the empirical slop.
        meter.append('misc/slop', misc.calc_slop(x, scores), type='max')

        # Record the gradient norm w.r.t Generator
        if gp1_lambda > 0:
            norm_nabla_hatfx = gp1_norm
        else:
            with torch.enable_grad():
                scores = scores.detach().clone()
                scores.requires_grad_(True)
                loss_real, loss_fake = loss_fn(scores)
                loss = loss_real.sum() + loss_fake.sum()
                nabla_L = torch.autograd.grad(loss, scores)[0]
                nabla_L_scale = nabla_L.abs().view(-1)
                norm_nabla_hatfx = (
                    collector.norm_nabla_hatfx * bs_D / nabla_L_scale)
        meter.append('misc/norm_nabla_hatfx', norm_nabla_hatfx.mean().cpu())

        # Record upper/lower bound of gradient norm for GN-GAN only.
        if norm_nabla_fx is not None:
            """If normalize_D return norm_nabla_fx, it is gradient normalization."""
            norm_nabla_hatfx_lb = (1 + x.flatten(start_dim=1).norm(dim=1))
            norm_nabla_hatfx_lb = norm_nabla_hatfx_lb.square().reciprocal().mean()
            margin = norm_nabla_hatfx - norm_nabla_hatfx_lb
            meter.append('misc/norm_nabla_hatfx_lb', norm_nabla_hatfx_lb.cpu())
            meter.append('misc/norm_nabla_fx', norm_nabla_fx.mean().cpu())
            meter.append('misc/margin', margin.cpu())


def train_G(
    meter: misc.Meter,          # Meter for recording loss value.
    D: torch.nn.Module,         # Discriminator.
    G: torch.nn.Module,         # Generator.
    loss_fn: callable,          # Loss function.
    normalize_fn: callable,     # Normalization function for G.
    bs_G: int,                  # Batch size for G.
    n_classes: int,             # Number of classes in the dataset.
    z_dim: int,                 # Dimension of latent space.
    **kwargs,
):
    device = dist.device()
    z = torch.randn(bs_G, z_dim, device=device)
    y = torch.randint(n_classes, (bs_G,), device=device)
    _, loss_G, _ = normalize_fn(D, G(z, y), loss_fn, y=y)
    loss_G.mean().backward()

    meter.append('loss/G', loss_G.mean().detach().cpu())


def training_loop(
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
    normalize_D: str,       # Normalization function for D.
    normalize_G: str,       # Normalization function for G.
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
    cr_lambda: float,       # Consistency regularization gamma.
    gp0_lambda: float,      # 0 Gradient penalty gamma.
    gp1_lambda: float,      # 1 Gradient penalty gamma.
    rescale_alpha: float,   # Alpha parameter of the rescaling.
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
    assert bs_D % (accumulation * dist.num_gpus()) == 0, "bs_D is not divisible by (accumulation * num_gpus)"
    assert bs_G % (accumulation * dist.num_gpus()) == 0, "bs_G is not divisible by (accumulation * num_gpus)"
    bs_D = bs_D // (accumulation * dist.num_gpus())
    bs_G = bs_G // (accumulation * dist.num_gpus())
    misc.set_seed(dist.rank() + seed)

    device = dist.device()
    dataset = datasets.Dataset(data_path, hflip, resolution, cr_lambda > 0)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, seed=seed, drop_last=True)
    else:
        sampler = RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs_D,
        sampler=sampler,
        num_workers=min(torch.get_num_threads(), 16),
        drop_last=True)
    loader = infiniteloop(loader, sampler)

    # Construct Models.
    D = misc.dynamic_import(architecture_D)(resolution, n_classes).to(device)
    G = misc.dynamic_import(architecture_G)(resolution, n_classes, z_dim).to(device)
    G_ema = copy.deepcopy(G)

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
    loss_fn_D = misc.dynamic_import(loss_D)
    loss_fn_G = misc.dynamic_import(loss_G)

    # The abstraction of normalization function. It is compatible with vanilla GAN
    normalize_fn_D = misc.dynamic_import(normalize_D)
    normalize_fn_G = misc.dynamic_import(normalize_G)

    # tf board writer.
    if dist.is_main():
        writer = SummaryWriter(logdir)
        # Collect following information for each layer:
        # - the norm of forward propagation
        # - the norm of gradient w.r.t parameter
        if isinstance(D, DistributedDataParallel):
            collector = misc.ForwAndParamGradCollector(D.module)
        else:
            collector = misc.ForwAndParamGradCollector(D)

    if not resume:
        # Sample fixed random noises and classes.
        fixed_z = torch.randn(sample_size, z_dim, device=device)
        fixed_z = fixed_z.chunk(dist.num_gpus(), dim=0)
        fixed_y = torch.randint(n_classes, (sample_size,), device=device)
        fixed_y = fixed_y.chunk(dist.num_gpus(), dim=0)
        # Initialize iteration and best results.
        start_step = 0
        best = {
            'FID/best': float('inf'),
            'FID/ema/best': float('inf'),
        }
        if dist.is_main():
            # Save arguments to `config.json`.
            with open(os.path.join(logdir, "config.json"), 'w') as f:
                json.dump(kwargs, f, indent=2, sort_keys=True)
            samples = [(dataset[i][0] + 1) / 2 for i in range(sample_size)]
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
        fixed_z = fixed_z.chunk(dist.num_gpus(), dim=0)
        fixed_y = ckpt['fixed_y'].to(device)
        fixed_y = fixed_y.chunk(dist.num_gpus(), dim=0)
        start_step = ckpt['step']
        best = ckpt['best']
        del ckpt

    # Initialize models for multi-gpu training.
    if dist.is_initialized():
        D = DistributedDataParallel(D, device_ids=[dist.device()])
        G = SyncBatchNorm.convert_sync_batchnorm(G)
        G = DistributedDataParallel(G, device_ids=[dist.device()])
        G_ema = SyncBatchNorm.convert_sync_batchnorm(G_ema)
        G_ema = DistributedDataParallel(G_ema, device_ids=[dist.device()])
    # This must be done after the initialization of DDP.
    G_ema.requires_grad_(False)

    # This must be done after the config file is saved.
    kwargs['bs_D'] //= (accumulation * dist.num_gpus())
    kwargs['bs_G'] //= (accumulation * dist.num_gpus())

    progress = trange(
        start_step + 1,         # Initial step value.
        steps + 1,              # The value is from 1 to steps (include).
        initial=start_step,     # Initial progress value.
        total=steps,            # The progress size.
        ncols=0,                # Disable bar, only show steps and percentage.
        desc='Training',
        disable=not dist.is_main())

    for step in progress:
        meter = misc.Meter()

        # Update D.
        D.requires_grad_(True)
        for _ in range(step_D):
            if rescale_alpha is not None:
                if isinstance(D, DistributedDataParallel):
                    D.module.rescale(alpha=rescale_alpha)
                else:
                    D.rescale(alpha=rescale_alpha)
            for i in range(accumulation):
                with dist.ddp_sync(D, sync=(i == accumulation - 1)):
                    train_D(loader, meter, D, G, loss_fn_D, normalize_fn_D, **kwargs)
            if accumulation != 1:
                for param in D.parameters():
                    param.grad.div_(accumulation)
            D_opt.step()
            D_opt.zero_grad(set_to_none=True)
        D_lrsched.step()
        D.requires_grad_(False)

        # Record the last forward pass of D.
        if dist.is_main():
            for tag, value in collector.norms():
                writer.add_scalar(tag, value, step)

        # Update G.
        G.requires_grad_(True)
        for i in range(accumulation):
            with dist.ddp_sync(G, sync=(i == accumulation - 1)):
                train_G(meter, D, G, loss_fn_G, normalize_fn_G, **kwargs)
        if accumulation != 1:
            for param in G.parameters():
                param.grad.div_(accumulation)
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

        # Update tf board and progress bar
        if dist.is_main():
            losses = meter.todict()
            for tag, value in losses.items():
                writer.add_scalar(tag, value, step)

            progress.set_postfix_str(", ".join([
                f"D_fake: {losses['loss/D/fake']:.3f}",
                f"D_real: {losses['loss/D/real']:.3f}",
                f"G: {losses['loss/G']:.3f}",
            ]))

        # Generate images from fixed z every sample_step steps.
        if step == 1 or step % sample_step == 0:
            with torch.no_grad():
                imgs = G(fixed_z[dist.rank()], fixed_y[dist.rank()])
                imgs = (imgs + 1) / 2
                imgs_ema = G_ema(fixed_z[dist.rank()], fixed_y[dist.rank()])
                imgs_ema = (imgs_ema + 1) / 2
            imgs = dist.gather(imgs)
            imgs_ema = dist.gather(imgs_ema)
            if dist.is_main():
                writer.add_image(
                    'fake', torchvision.utils.make_grid(imgs.cpu()), step)
                writer.add_image(
                    'fake/ema', torchvision.utils.make_grid(imgs_ema.cpu()), step)

        # Calculate FID every eval_step steps.
        if step == 1 or step % eval_step == 0:
            FID = fid(G, **kwargs)
            FID_ema = fid(G_ema, **kwargs)
            if dist.is_main():
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
                if dist.is_initialized():
                    ckpt = {
                        'G': G.module.state_dict(),
                        'D': D.module.state_dict(),
                        'G_ema': G_ema.module.state_dict(),
                    }
                else:
                    ckpt = {
                        'G': G.state_dict(),
                        'D': D.state_dict(),
                        'G_ema': G_ema.state_dict(),
                    }
                ckpt.update({
                    'G_opt': G_opt.state_dict(),
                    'D_opt': D_opt.state_dict(),
                    'G_lrsched': G_lrsched.state_dict(),
                    'D_lrsched': D_lrsched.state_dict(),
                    'fixed_z': torch.cat(fixed_z, dim=0),
                    'fixed_y': torch.cat(fixed_y, dim=0),
                    'best': best,
                    'step': step,
                })
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

    if dist.is_main():
        progress.close()
        writer.close()
