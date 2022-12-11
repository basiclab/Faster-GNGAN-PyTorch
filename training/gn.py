import torch


class GradUpdateHook:
    """Faster GN-GAN"""
    def __init__(self, f, x, loss_fn):
        self.f = f[:, 0]        # [B, 1] -> [B]
        self.loss_fn = loss_fn
        self.handle = x.register_hook(self)

    def loss_scale(self, grad_norm):
        f_hat = self.f / (grad_norm + torch.abs(self.f))
        with torch.enable_grad():
            f_hat.requires_grad_(True)
            loss = self.loss_fn(f_hat).sum()
            scale = torch.autograd.grad(loss, f_hat)[0]
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


def normalize_D(D, x, loss_fn, **kwargs):
    """
    Apply Gradient Normalization on D with input x.
                     f
    f_hat = -------------------
            || df/dx || + | f |

    Returns:
        f_hat: discriminator output
        loss: loss
    """
    x.requires_grad_(True)
    f = D(x, **kwargs)
    grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
    grad_norm = grad.flatten(start_dim=1).norm(dim=1).view(-1, 1)
    f_hat = f / (grad_norm + torch.abs(f))
    loss = loss_fn(f_hat)
    return f_hat, loss, grad_norm


def normalize_G(D, x, loss_fn, **kwargs):
    """
    Returns:
        f: discriminator output
        loss: the mean of f
    """
    f = D(x, **kwargs)
    GradUpdateHook(f, x, loss_fn)
    loss = f
    return f, loss, None


def vanilla_D(D, x, loss_fn, **kwargs):
    f = D(x, **kwargs)
    loss = loss_fn(f)
    return f, loss, None


def vanilla_G(D, x, loss_fn, **kwargs):
    f = D(x, **kwargs)
    loss = loss_fn(f)
    return f, loss, None
