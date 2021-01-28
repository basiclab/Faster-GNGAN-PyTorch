import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = super().forward(
                self.scale * pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(
                self.scale * pred_fake, torch.zeros_like(pred_fake))
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = super().forward(
                self.scale * pred_fake, torch.ones_like(pred_fake))
            return loss


class HingeLoss(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, pred_fake, pred_real=None):
        if pred_real is not None:
            loss_real = F.relu(1 - self.scale * pred_real).mean()
            loss_fake = F.relu(1 + self.scale * pred_fake).mean()
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_fake.mean()
            return loss


loss_fns = {
    'bce': BCEWithLogits,
    'hinge': HingeLoss,
}
