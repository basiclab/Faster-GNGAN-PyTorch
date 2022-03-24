import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def forward(self, pred_fake, pred_real=None):
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

    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = self.bce(pred_real, torch.ones_like(pred_real))
            loss_fake = self.bce(pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_fake, loss_real
        else:
            loss = self.bce(pred_fake, torch.ones_like(pred_fake))
            return loss


class HingeLoss(BaseLoss):
    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss, loss_fake, loss_real
        else:
            loss = -pred_fake.mean()
            return loss


class Wasserstein(BaseLoss):
    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = -loss_real + loss_fake
            return loss, loss_fake, loss_real
        else:
            loss = -pred_fake.mean()
            return loss


class BCE(BaseLoss):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = self.bce(
                (pred_real + 1) / 2, torch.ones_like(pred_real))
            loss_fake = self.bce(
                (pred_fake + 1) / 2, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_fake, loss_real
        else:
            loss = self.bce(
                (pred_fake + 1) / 2, torch.ones_like(pred_fake))
            return loss
