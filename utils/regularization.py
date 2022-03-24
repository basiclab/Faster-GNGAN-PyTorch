import torch
import torchvision.transforms as transforms


from models.gradnorm import normalize_gradient_D


def consistency_loss(net_D, real, y_real, pred_real,
                     transform=transforms.Compose([
                        transforms.Lambda(lambda x: (x + 1) / 2),
                        transforms.ToPILImage(mode='RGB'),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(0, translate=(0.2, 0.2)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ])):
    aug_real = real.detach().clone().cpu()
    for idx, img in enumerate(aug_real):
        aug_real[idx] = transform(img)
    aug_real = aug_real.to(real.device)
    pred_aug = normalize_gradient_D(net_D, aug_real, y=y_real)
    loss = ((pred_aug - pred_real) ** 2).mean()
    return loss
