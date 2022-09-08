import contextlib
import importlib
import json
import random

import click
import numpy as np
import torch


class CommandAwareConfig(click.Command):
    def invoke(self, ctx):
        """Load config from file and overwrite by command line arguments."""
        config_file = ctx.params["config"]
        if config_file is None:
            return super(CommandAwareConfig, self).invoke(ctx)
        with open(config_file) as f:
            configs = json.load(f)
        for param in ctx.params.keys():
            if ctx.get_parameter_source(param) != click.core.ParameterSource.DEFAULT:
                continue
            if param in configs:
                ctx.params[param] = configs[param]
        return super(CommandAwareConfig, self).invoke(ctx)


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
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


class Collector(object):
    def __init__(self, module):
        self.hooks = dict()

        for name, m in module.named_modules():
            if len(m._parameters) > 0:
                forward_key = f"norm/forward/1/{name}"
                self.hooks[forward_key] = self.ForwardHook()
                m.register_forward_hook(self.hooks[forward_key])

        for name, p in module.named_parameters():
            key = f"norm/grad/{name}"
            self.hooks[key] = self.GradientHook()
            p.register_hook(self.hooks[key])

            key = f"norm/param/{name}"
            self.hooks[key] = self.WeightHook(p, key)
            p.register_hook(self.hooks[key])

    class ForwardHook:
        """Collect the average norm of the output tensor."""
        def __init__(self):
            self.norm = None

        @torch.no_grad()
        def __call__(self, module, inputs, outputs):
            # print('ForwardHook:', outputs.shape, self.norm, flush=True)
            self.norm = outputs.clone().flatten(start_dim=1).norm(dim=1).mean()

    class GradientHook:
        """Collect the gradient norm w.r.t the parameter."""
        def __init__(self):
            self.norm = None

        @torch.no_grad()
        def __call__(self, grad):
            # print('GradientHook', grad.shape, self.norm, flush=True)
            self.norm = grad.norm()

    class WeightHook:
        """Collect the weight norm at backward."""
        def __init__(self, param, key):
            self.norm = None
            self.param = param
            self.key = key

        @torch.no_grad()
        def __call__(self, grad):
            # print('WeightHook:', self.param.shape, self.key, self.norm, flush=True)
            self.norm = self.param.norm()

    def norms(self):
        for tag, hook in self.hooks.items():
            if hook.norm is not None:
                yield tag, hook.norm


class GradFxCollector(object):
    def __init__(self, x: torch.Tensor):
        self.handle = x.register_hook(self)
        self.norm = None

    @torch.no_grad()
    def __call__(self, grad: torch.Tensor):
        self.norm = grad.flatten(start_dim=1).norm(dim=1).mean()
        self.handle.remove()


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


if __name__ == '__main__':
    from training.models.dcgan import Discriminator
    from training.losses import WGANLoss
    from training.gn import normalize_D

    loss_fn = WGANLoss()
    D = Discriminator(32, None)
    collector = Collector(D)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = normalize_D(D, x, None)
    y.mean().backward()
