import contextlib

import torch
import torch.distributed as dist


def is_initialized():
    return dist.is_initialized()


def rank() -> int:
    if is_initialized():
        return dist.get_rank()
    else:
        return 0


def device() -> torch.device:
    return torch.device('cuda', rank())


def num_gpus():
    if is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_main() -> int:
    return rank() == 0


def barrier():
    if is_initialized():
        dist.barrier()


def gather(tensor: torch.Tensor):
    """Concate tensors from all processes along the first dimension."""
    if is_initialized():
        if is_main():
            gather_list = [
                tensor.new_empty(tensor.shape)
                for _ in range(dist.get_world_size())
            ]
        else:
            gather_list = None
        dist.gather(tensor, gather_list)
        if is_main():
            tensor = torch.cat(gather_list, dim=0)
            return tensor   # rank 0
        else:
            return None     # other ranks
    else:
        return tensor


@contextlib.contextmanager
def ddp_sync(module, sync):
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield
