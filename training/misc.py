import contextlib
import importlib
import json
import random

import click
import numpy as np
import torch
import torchvision


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


def collect_forward_backward_norm(module: torch.nn.Module):
    class Collector:
        class ForwardHook:
            """Collect the average norm of the output tensor."""
            def __init__(self):
                self.norm = None

            @torch.no_grad()
            def __call__(self, module, inputs, outputs):
                self.norm = outputs.flatten(start_dim=1).norm(dim=1).sum()
                # print('FP:', outputs.shape, self.norm)

        class BackwardHook:
            """Collect the average gradient norm w.r.t the input tensor."""
            def __init__(self):
                self.norm = None

            @torch.no_grad()
            def __call__(self, module, grad_input, grad_output):
                if grad_input[0] is None:
                    return
                self.norm = grad_input[0].flatten(start_dim=1).norm(dim=1).sum()
                # print('BP:', grad_input[0].shape, self.norm)

        class GradientHook:
            """Collect the gradient norm w.r.t the parameter."""
            def __init__(self):
                self.norm = None

            @torch.no_grad()
            def __call__(self, grad):
                self.norm = grad.norm()
                # print('BP w.r.t Parameter:', grad.shape, self.norm)

        def __init__(self, module):
            self.module = module
            self.hooks = dict()
            for name, m in module.named_modules():
                if len(m._parameters) > 0:
                    forward_key = f"norm/forward/{name}"
                    self.hooks[forward_key] = self.ForwardHook()
                    m.register_forward_hook(self.hooks[forward_key])
                    # backward_key = f"norm/backward/{name}"
                    # self.hooks[backward_key] = self.BackwardHook()
                    # m.register_full_backward_hook(self.hooks[backward_key])
            for name, p in module.named_parameters():
                key = f"norm/grad/{name}"
                self.hooks[key] = self.GradientHook()
                p.register_hook(self.hooks[key])

        def norms(self):
            for tag, hook in self.hooks.items():
                if hook.norm is not None:
                    yield tag, hook.norm
    return Collector(module)


@contextlib.contextmanager
def ddp_sync(module, sync):
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


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


cr_augment = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(0, translate=(0.2, 0.2)),
])


if __name__ == '__main__':
    from .models import resnet

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    m = resnet.Discriminator(32, 1)
    collector = collect_forward_backward_norm(m)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    y = m(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    y = y.div(grad.flatten(1).norm(dim=1).add(y.norm(dim=1)).square())
    y = y.sum()
    print("====")
    y.sum().backward()
    for tag, norm in collector.norms():
        print(tag, norm)
