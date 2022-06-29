import torch


class Hook(object):
    def __init__(self, f, loss_fn, use_fn):
        self.f = f[:, 0]        # [B, 1] -> [B]
        self.loss_fn = loss_fn
        self.use_fn = use_fn
        self.handle = None

    def set_handle(self, handle):
        self.handle = handle

    def loss_scale(self, grad_norm):
        f_hat = self.f / (grad_norm + torch.abs(self.f))
        with torch.enable_grad():
            f_hat.requires_grad_(True)
            loss = self.loss_fn(f_hat)
            scale = torch.autograd.grad(loss, f_hat)[0] * self.f.shape[0]
        return scale.view(-1, 1, 1, 1)

    def grad_scale(self, grad_norm):
        if self.use_fn:
            scale = grad_norm / ((grad_norm + torch.abs(self.f)) ** 2)
        else:
            scale = 1 / grad_norm
        return scale.view(-1, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, grad):
        """
        dL     dL           || df/dx ||           df
        -- = ------ x ----------------------- x ------
        dx   df_hat   (|| df/dx || + | f |)^2     dx
             ~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~
               |                 |                |
               v                 v                v
           loss scale     gradient scale   original gradient
        """
        grad_norm = torch.norm(
            torch.flatten(grad, start_dim=1), p=2, dim=1) * grad.shape[0]
        grad = self.loss_scale(grad_norm) * self.grad_scale(grad_norm) * grad
        self.handle.remove()
        return grad


def normalize_G(net_D, x, loss_fn, use_fn, **kwargs):
    f = net_D(x, **kwargs)
    hook = Hook(f, loss_fn, use_fn)
    handle = x.register_hook(hook)
    hook.set_handle(handle)
    return f


def normalize_D(net_D, x, loss_fn, use_fn, **kwargs):
    """
                     f
    f_hat = -------------------
            || df/dx || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
    grad_norm = grad.flatten(start_dim=1).norm(dim=1).view(-1, 1)
    if use_fn:
        f_hat = f / (grad_norm + torch.abs(f))
    else:
        f_hat = f / grad_norm
    return f_hat
