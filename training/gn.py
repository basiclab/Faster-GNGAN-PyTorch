import torch


class GradUpdateHook(object):
    def __init__(self, f, x, loss_fn):
        self.f = f[:, 0]        # [B, 1] -> [B]
        self.loss_fn = loss_fn
        self.handle = x.register_hook(self)

    def loss_scale(self, grad_norm):
        f_hat = self.f / (grad_norm + torch.abs(self.f))
        with torch.enable_grad():
            f_hat.requires_grad_(True)
            loss = self.loss_fn(f_hat)
            scale = torch.autograd.grad(loss, f_hat)[0] * self.f.shape[0]
        return scale.view(-1, 1, 1, 1)

    def grad_scale(self, grad_norm):
        scale = grad_norm / ((grad_norm + torch.abs(self.f)) ** 2)
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
        # this is required to avoid memory leak
        self.handle.remove()
        return grad


def normalize_G(D, x, loss_fn, **kwargs):
    f = D(x, **kwargs)
    GradUpdateHook(f, x, loss_fn)
    return f, None


def normalize_D(D, x, loss_fn, **kwargs):
    """
                     f
    f_hat = -------------------
            || df/dx || + | f |
    """
    x.requires_grad_(True)
    f = D(x, **kwargs)
    grad = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
    grad_norm = grad.flatten(start_dim=1).norm(dim=1).view(-1, 1)
    f_hat = f / (grad_norm + torch.abs(f))
    return f_hat, grad_norm
