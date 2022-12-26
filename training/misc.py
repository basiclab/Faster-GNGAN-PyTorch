import importlib
import json
import random
from collections import defaultdict

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
            if param != "config" and param in configs:
                ctx.params[param] = configs[param]
        return super(CommandAwareConfig, self).invoke(ctx)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def dynamic_import(module):
    module, class_name = module.rsplit('.', maxsplit=1)
    return getattr(importlib.import_module(module), class_name)


def calc_slop(x, y):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    diff_x = torch.linalg.vector_norm(x[:, None, :] - x[None, :, :], dim=2)
    diff_y = torch.linalg.vector_norm(y[:, None, :] - y[None, :, :], dim=2)
    slop = torch.max(diff_y / (diff_x + 1e-12))

    return slop


class ForwAndParamGradCollector(object):
    def __init__(self, module):
        self.hooks = dict()

        for name, m in module.named_modules():
            if len(m._parameters) > 0:
                forward_key = f"forward/norm/{name}"
                self.hooks[forward_key] = self.ForwardHook()
                m.register_forward_hook(self.hooks[forward_key])

        for name, p in module.named_parameters():
            key = f"param/grad/norm/{name}"
            self.hooks[key] = self.GradientHook()
            p.register_hook(self.hooks[key])

            key = f"param/norm/{name}"
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


class NablaHatFxCollector(object):
    def __init__(self, module: torch.nn.Module):
        self.handle_forward = module.register_forward_pre_hook(self.forward_hook)
        self.handle_backward = module.register_full_backward_hook(self.backward_hook)
        self.norm_nabla_hatfx = None
        self.forward_flag = False

    @torch.no_grad()
    def forward_hook(self, module, input):
        # Record the first forward pass only.
        if not self.forward_flag:
            input[0].requires_grad_(True)
            self.forward_flag = True

    @torch.no_grad()
    def backward_hook(self, module, input_grad, output_grad):
        self.norm_nabla_hatfx = input_grad[0].flatten(start_dim=1).norm(dim=1)

    def remove(self):
        self.handle_forward.remove()
        self.handle_backward.remove()


class Meter:
    def __init__(self):
        self.nametype2data = defaultdict(list)

    def append(self, name: str, value: torch.Tensor, type: str = 'mean'):
        self.nametype2data[(name, type)].append(value)

    def todict(self):
        result = dict()
        for (name, type), value_lst in self.nametype2data.items():
            if type == 'mean':
                value = torch.mean(torch.stack(value_lst))
            elif type == 'max':
                value = torch.max(torch.stack(value_lst))
            else:
                raise ValueError(f'Unknown type: {type}')
            result[name] = value
        return result


if __name__ == '__main__':
    from training.models.dcgan import Discriminator
    from training.losses import wgan_loss_D
    from training.gn import normalize_D

    D = Discriminator(32, None)
    collector = Collector(D)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y, _, _ = normalize_D(D, x, wgan_loss_D)
    y.mean().backward()
