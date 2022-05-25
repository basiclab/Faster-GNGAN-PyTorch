import torch.nn as nn
import torch.nn.functional as F


class NSLoss(nn.Module):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = F.softplus(-scores_real).mean()
            loss_fake = F.softplus(scores_fake).mean()
            return loss_fake, loss_real
        else:
            loss = F.softplus(-scores_fake).mean()
            return loss


class HingeLoss(nn.Module):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = F.relu(1 - scores_real).mean()
            loss_fake = F.relu(1 + scores_fake).mean()
            return loss_fake, loss_real
        else:
            loss = -scores_fake.mean()
            return loss


class WGANLoss(nn.Module):
    def forward(self, scores_fake, scores_real=None):
        if scores_real is not None:
            loss_real = -scores_real.mean()
            loss_fake = scores_fake.mean()
            return loss_fake, loss_real
        else:
            loss = -scores_fake.mean()
            return loss
