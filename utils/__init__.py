import random
from contextlib import contextmanager

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def infiniteloop(dataloader, sampler, step):
    epoch = step // len(dataloader)    # resume training from last epoch
    while True:
        sampler.set_epoch(epoch)
        for x, y in iter(dataloader):
            yield x, y
        epoch += 1


def ema(source, target, decay):
    """
    Args:
        source (torch.Tensor): source training model.
        target (torch.Tensor): target model for caching averaged parameters.
        decay (float): the decay rate of averaged parameters.
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])


def generate_images(net_G, batch_size, num_images, z_dim, n_classes,
                    verbose=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = batch_size // world_size
    with torch.no_grad():
        if rank == 0:
            progress = tqdm(
                total=num_images, desc="Generate Imgs", ncols=0,
                leave=False, disable=(verbose == 0))
        for iter_i in range(0, num_images, batch_size):
            z = torch.randn(local_batch_size, z_dim).to(rank)
            y = torch.randint(n_classes, (local_batch_size,)).to(rank)
            fake = net_G(z, y)
            buffer = [torch.empty_like(fake) for _ in range(world_size)]
            dist.all_gather(buffer, fake)
            if rank == 0:
                fake = torch.cat(buffer, dim=0)[:num_images - iter_i]
                yield fake
                progress.update(len(fake))
            else:
                yield None
            dist.barrier()
        if rank == 0:
            progress.close()
