import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            loss = loss_real + loss_fake
            return loss, loss_real, loss_fake
        else:
            loss = -pred_real.mean()
            return loss
