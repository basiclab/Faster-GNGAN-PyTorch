import torch
import torch.nn.functional as F


def ns_loss_D(scores):
    scores_real, scores_fake = scores.chunk(2, dim=0)
    loss_real = F.softplus(-scores_real)
    loss_fake = F.softplus(scores_fake)
    return loss_real, loss_fake


def ns_loss_G(scores):
    loss = F.softplus(-scores)
    return loss


def bce_loss_D(scores):
    """scores are in range [-1, 1]"""
    scores = (scores + 1) / 2
    scores_real, scores_fake = scores.chunk(2, dim=0)
    loss_real = F.binary_cross_entropy(
        scores_real, torch.ones_like(scores_real), reduction='none')
    loss_fake = F.binary_cross_entropy(
        scores_fake, torch.zeros_like(scores_fake), reduction='none')
    return loss_real, loss_fake


def bce_loss_G(scores):
    loss = F.binary_cross_entropy(
        scores, torch.ones_like(scores), reduction='none')
    return loss


def hinge_loss_D(scores):
    scores_real, scores_fake = scores.chunk(2, dim=0)
    loss_real = F.relu(1 - scores_real)
    loss_fake = F.relu(1 + scores_fake)
    return loss_real, loss_fake


def hinge_loss_G(scores):
    loss = -scores
    return loss


def wgan_loss_D(scores):
    scores_real, scores_fake = scores.chunk(2, dim=0)
    loss_real = -scores_real
    loss_fake = scores_fake
    return loss_fake, loss_real


def wgan_loss_G(scores):
    loss = -scores
    return loss
