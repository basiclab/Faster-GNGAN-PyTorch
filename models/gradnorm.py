from functools import partial

import torch
import torch.nn as nn


def normalize_gradient_G(net_D, loss_fn, x, **kwargs):
    def hook(grad, f, loss_fn):
        """
        This term is equivelent to grad_scale * grad
                         ^
                         |
                      ~~~~~~
        dL     dL     df_hat
        -- = ------ x ------
        dx   df_hat     dx
             ~~~~~~
                |
                v
        See losses.BaseLoss for details
        """
        with torch.no_grad():
            grad_norm = torch.norm(
                torch.flatten(grad, start_dim=1), p=2, dim=1) * grad.shape[0]
            f = f[:, 0]
            grad_scale = grad_norm / ((grad_norm + torch.abs(f)) ** 2)
            grad_scale = grad_scale.view(-1, 1, 1, 1)
            loss_scale = loss_fn.get_scale(f, grad_norm).view(-1, 1, 1, 1)
            grad = loss_scale * grad_scale * grad
        return grad
    f = net_D(x, **kwargs)
    h = x.register_hook(partial(hook, f=f, loss_fn=loss_fn))
    return f, h


def normalize_gradient_D(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, 1)
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat


class Rescalable(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        assert 'weight' in module._parameters

    def init_module_scale(self):
        if 'weight' in self.module._parameters:
            self.init_param_scale('weight')
        if 'bias' in self.module._parameters:
            self.init_param_scale('bias')

    def init_param_scale(self, name):
        params = self.module._parameters[name]
        self.module.register_parameter(f"{name}_raw", params)
        self.module.register_buffer(f'{name}_scale', torch.ones(()))
        self.module.register_buffer(f'{name}_norm', params.data.norm(p=2))
        delattr(self.module, name)
        setattr(self.module, name, params.data)

    @torch.no_grad()
    def rescale(self, base_scale=1.):
        if 'weight_raw' in self.module._parameters:
            self.module.weight_scale = self.module.weight_norm / (
                self.module.weight_raw.norm(p=2) + 1e-12)
            base_scale = base_scale * self.module.weight_scale
        if 'bias_raw' in self.module._parameters:
            self.module.bias_scale = base_scale
        return base_scale

    def forward(self, *args, **kwargs):
        for name in ['weight', 'bias']:
            if f"{name}_raw" in self.module._parameters:
                param = self.module._parameters[f"{name}_raw"]
                scale = self.module._buffers[f'{name}_scale']
                setattr(self.module, name, param * scale)
        return self.module(*args, **kwargs)
