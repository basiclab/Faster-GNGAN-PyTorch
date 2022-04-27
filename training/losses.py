import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def forward(self, scores_fake, scores_real=None):
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


class NSLoss(BaseLoss):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = F.softplus(-scores_real).mean()
            loss_fake = F.softplus(scores_fake).mean()
            return loss_fake, loss_real
        else:
            loss = F.softplus(-scores_fake).mean()
            return loss


class HingeLoss(BaseLoss):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = F.relu(1 - scores_real).mean()
            loss_fake = F.relu(1 + scores_fake).mean()
            return loss_fake, loss_real
        else:
            loss = -scores_fake.mean()
            return loss


class WGANLoss(BaseLoss):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = scores_real.mean()
            loss_fake = scores_fake.mean()
            return loss_fake, loss_real
        else:
            loss = -scores_fake.mean()
            return loss
