from functools import partial

import torch


def normalize_gradient_G(net_D, x, **kwargs):
    def hook(grad, f):
        with torch.no_grad():
            grad_norm = torch.norm(
                torch.flatten(grad, start_dim=1), p=2, dim=1) * grad.shape[0]
            f = f[:, 0]
            scale = grad_norm / ((grad_norm + torch.abs(f)) ** 2)
            scale = scale.view(-1, 1, 1, 1)
        grad = grad * scale
        return grad
    f = net_D(x, **kwargs)
    x.register_hook(partial(hook, f=f))
    return f


def normalize_gradient_D(net_D, x, **kwargs):
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f),
        create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, 1)
    f = (f / (grad_norm + torch.abs(f)))
    return f
