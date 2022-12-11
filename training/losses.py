import torch.nn.functional as F


def ns_loss_D(scores):
    scores_real, scores_fake = scores.chunk(2, dim=0)
    loss_real = F.softplus(-scores_real)
    loss_fake = F.softplus(scores_fake)
    return loss_real, loss_fake


def ns_loss_G(scores):
    loss = F.softplus(-scores)
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
