import contextlib
import importlib
import json
import random

import click
import numpy as np
import torch
import torchvision


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def construct_class(module, *args, **kwargs):
    module, class_name = module.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), class_name)(*args, **kwargs)


@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


def CommandAwareConfig(config_param_name):
    class CustomCommandClass(click.Command):
        def invoke(self, ctx):
            config_file = ctx.params[config_param_name]
            if config_file is None:
                return super(CustomCommandClass, self).invoke(ctx)
            with open(config_file) as f:
                configs = json.load(f)
            for param in ctx.params.keys():
                if ctx.get_parameter_source(param) != click.core.ParameterSource.DEFAULT:
                    continue
                if param in configs:
                    ctx.params[param] = configs[param]
            return super(CustomCommandClass, self).invoke(ctx)
    return CustomCommandClass


class Meter:
    def __init__(self):
        self.name2list = dict()

    def append(self, name: str, value: torch.Tensor):
        value_lst = self.name2list.get(name, [])
        value_lst.append(value)
        self.name2list[name] = value_lst

    def todict(self):
        return {
            name: torch.mean(torch.stack(value_lst))
            for name, value_lst in self.name2list.items()
        }


class LinearLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, end_factor=0.0, total_steps=5):
        self.end_factor = end_factor
        self.total_steps = total_steps
        super(LinearLR, self).__init__(optimizer, self)

    def __call__(self, step):
        return 1 - (1 - self.end_factor) * (step / self.total_steps)


cr_augment = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(0, translate=(0.2, 0.2)),
])
