def test_model_rescaling():
    import torch
    from training.models import resnet, dcgan, biggan

    Models = [
        (32, 10, biggan.Discriminator, 'BigGAN'),
        (128, 1000, biggan.Discriminator, 'BigGAN'),
        (32, 1, resnet.Discriminator, 'ResNet'),
        (48, 1, resnet.Discriminator, 'ResNet'),
        (128, 1, resnet.Discriminator, 'ResNet'),
        (256, 1, resnet.Discriminator, 'ResNet'),
        (32, 1, dcgan.Discriminator, 'DCGAN'),
        (48, 1, dcgan.Discriminator, 'DCGAN'),
    ]
    alpha_range = torch.linspace(start=1, end=1e-2, steps=10)
    for res, n_classes, Model, family in Models:
        print("=" * 80)
        print(f"{family}(resolution={res}, n_classes={n_classes})")
        x = torch.randn(1, 3, res, res, requires_grad=True).cuda()
        y = torch.randint(n_classes, (1,)).cuda()
        net_D = Model(res, n_classes).cuda()
        f = net_D(x, y=y)
        grad_f = torch.autograd.grad(f.sum(), x)[0]
        grad_norm = torch.norm(torch.flatten(grad_f, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, 1)
        f_hat = f / (grad_norm + torch.abs(f))
        print(f'   '
              f'{"Output":>12s}, '
              f'{"Raw Output":>12s}, '
              f'{"Grad Norm":>12s}, '
              f'{"alpha":>7s};')
        print(f'   '
              f'{f_hat.item():+12.7f}, '
              f'{f.item():+12.7f}, '
              f'{grad_norm.item():+12.7f}')

        # Test with different alpha
        for step, alpha in enumerate(alpha_range):
            net_D.rescale(alpha=alpha)
            f_scaled = net_D(x, y=y)
            grad_f_scaled = torch.autograd.grad(f_scaled.sum(), x)[0]
            grad_norm_scaled = torch.norm(
                torch.flatten(grad_f_scaled, start_dim=1), p=2, dim=1)
            grad_norm_scaled = grad_norm_scaled.view(-1, 1)
            f_hat_scaled = f_scaled / (grad_norm_scaled + torch.abs(f_scaled))

            print(f'{step:2d} '
                  f'{f_hat_scaled.item():+12.7f}, '
                  f'{f_scaled.item():+12.7f}, '
                  f'{grad_norm_scaled.item():+12.7f}, '
                  f'{alpha.item():+7.3f}', end='; ')

            ratio1 = f_scaled / f
            ratio2 = grad_norm_scaled / grad_norm
            if not torch.allclose(
                    ratio1, ratio2, rtol=1e-02, atol=1e-06):
                print(f'[Warning 1] '
                      f'{ratio1.item():+.7f} != '
                      f'{ratio2.item():+.7f}', end='; ')
            if not torch.allclose(
                    f_hat_scaled, f_hat, rtol=1e-03, atol=1e-06):
                print(f'[warning 2] '
                      f'{f_hat_scaled.item():+.7f} != '
                      f'{f_hat.item():+.7f}', end='; ')
            print('')


def test_loss():
    import copy
    import torch
    import random
    import numpy as np

    from training import gn
    from training import losses
    from training.models import resnet, dcgan, biggan

    Models = [
        (32, 10, biggan.Discriminator, 'BigGAN'),
        (128, 1000, biggan.Discriminator, 'BigGAN'),
        (32, 1, resnet.Discriminator, 'ResNet'),
        (48, 1, resnet.Discriminator, 'ResNet'),
        (128, 1, resnet.Discriminator, 'ResNet'),
        (256, 1, resnet.Discriminator, 'ResNet'),
        (32, 1, dcgan.Discriminator, 'DCGAN'),
        (48, 1, dcgan.Discriminator, 'DCGAN'),
    ]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss_fns = [
        losses.NSLoss(),
        losses.HingeLoss(),
        losses.WGANLoss(),
    ]

    device = torch.device('cuda')

    for res, n_classes, Model, family in Models:
        print("=" * 80)
        print(f"{family}(resolution={res}, n_classes={n_classes})")
        D1 = Model(res, n_classes).to(device)
        D2 = copy.deepcopy(D1)

        for loss_fn in loss_fns:
            print("Test", loss_fn.__class__.__name__)

            for _ in range(10):
                x = torch.randn(2, 3, res, res, requires_grad=True, device=device)
                y = torch.randint(n_classes, (1,), device=device)
                x.retain_grad()

                D1.zero_grad()
                y1 = gn.normalize_gradient_D(D1, x, y=y)
                loss = loss_fn(y1)
                loss.backward()
                grad1 = x.grad.detach().clone()
                x.grad.zero_()

                D2.zero_grad()
                y2 = gn.normalize_gradient_G(D2, loss_fn, x, y=y)
                loss = y2.mean()
                loss.backward()
                grad2 = x.grad.detach().clone()
                x.grad.zero_()

                if not torch.allclose(grad1, grad2, rtol=1e-3):
                    print("[Warning] Relative Err%.7f" % (grad1 - grad2).abs().div(grad2.abs()).mean())
            print("Finish")


if __name__ == '__main__':
    # test_model_rescaling()
    test_loss()
