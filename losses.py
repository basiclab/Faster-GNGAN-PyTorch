import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        raise NotImplementedError

    def get_scale(self, f, grad_norm):
        """
        dL/dx = dL/df_hat * df_hat/dx
                ~~~~~~~~~
                    |
                    v
            Calculate this term
        """
        f_hat = f / (grad_norm + torch.abs(f))
        with torch.enable_grad():
            f_hat.requires_grad_(True)
            loss = self.forward(f_hat)
            scale = torch.autograd.grad(loss, f_hat)[0] * f.shape[0]
        return scale


class BCEWithLogits(BaseLoss):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = self.bce(pred_real, torch.ones_like(pred_real))
            loss_fake = self.bce(pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = self.bce(pred_real, torch.ones_like(pred_real))
            return loss


class HingeLoss(BaseLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class Wasserstein(BaseLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = -loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class BCE(BaseLoss):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = self.bce(
                (pred_real + 1) / 2, torch.ones_like(pred_real))
            loss_fake = self.bce(
                (pred_fake + 1) / 2, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = self.bce(
                (pred_real + 1) / 2, torch.ones_like(pred_real))
            return loss


if __name__ == '__main__':
    import random
    import numpy as np
    from models.gradnorm import normalize_gradient_G, normalize_gradient_D

    def create_D():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(32, 1),
        )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    D1 = create_D()
    D2 = create_D()
    D2.load_state_dict(D1.state_dict())

    loss_fns = [
        BCEWithLogits(),
        HingeLoss(),
        Wasserstein(),
        BCE(),
    ]
    for loss_fn in loss_fns:
        print(loss_fn.__class__.__name__)

        for _ in range(10):
            x = torch.randn(4, 3, 32, 32, requires_grad=True)

            D1.zero_grad()
            y1 = normalize_gradient_D(D1, x)
            loss = loss_fn(y1)
            loss.backward()
            grad1 = x.grad.detach().clone()
            x.grad.zero_()

            D2.zero_grad()
            y2 = normalize_gradient_G(D2, loss_fn, x)
            loss = y2.mean()
            loss.backward()
            grad2 = x.grad.detach().clone()
            x.grad.zero_()

            print(torch.max(torch.abs(grad1 - grad2)))

            assert torch.allclose(grad1, grad2)
