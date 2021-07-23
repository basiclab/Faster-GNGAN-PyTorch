from functools import partial

import torch


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


@torch.no_grad()
def scale_module(module, base_scale=1., min_norm=1.0, max_norm=1.33):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        weight_scale = module.weight.norm(p=2)
        weight_scale = torch.clamp(
            weight_scale, min_norm, max_norm)
        base_scale = base_scale * weight_scale
        module.weight.data.div_(weight_scale)
        module.bias.data.div_(base_scale)

    return base_scale
