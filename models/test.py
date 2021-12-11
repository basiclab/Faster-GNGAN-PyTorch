import torch

from models import biggan, resnet, dcgan


if __name__ == '__main__':
    """
    The unit test of model rescaling
    """
    Models = [
        (32, 10, biggan.Discriminator32, 'BigGAN'),
        (128, 1000, biggan.Discriminator128, 'BigGAN'),
        (32, 1, resnet.ResDiscriminator32, 'ResNet'),
        (48, 1, resnet.ResDiscriminator48, 'ResNet'),
        (128, 1, resnet.ResDiscriminator128, 'ResNet'),
        (256, 1, resnet.ResDiscriminator256, 'ResNet'),
        (32, 1, dcgan.Discriminator32, 'DCGAN'),
        (48, 1, dcgan.Discriminator48, 'DCGAN'),
    ]
    alpha_range = torch.linspace(start=1, end=1e-2, steps=10)
    # Test each discriminator in list
    for res, n_classes, Model, family in Models:
        print("=" * 80)
        print(family, Model.__name__)
        x = torch.randn(1, 3, res, res, requires_grad=True).cuda()
        y = torch.randint(n_classes, (1,)).cuda()
        net_D = Model(n_classes).cuda()
        f = net_D(x, y=y)
        grad_f = torch.autograd.grad(f.sum(), x)[0]
        grad_norm = torch.norm(torch.flatten(grad_f, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, 1)
        f_hat = f / (grad_norm + torch.abs(f))
        print(f'   '
              f'{"Output":>11s}, '
              f'{"Raw Output":>11s}, '
              f'{"Grad Norm":>11s}, '
              f'{"alpha":>7s};')
        print(f'   '
              f'{f_hat.item():+11.7f}, '
              f'{f.item():+11.7f}, '
              f'{grad_norm.item():+11.7f}')

        # Test with different alpha
        for step, alpha in enumerate(alpha_range):
            net_D.rescale_model(alpha)
            f_scaled = net_D(x, y=y)
            grad_f_scaled = torch.autograd.grad(f_scaled.sum(), x)[0]
            grad_norm_scaled = torch.norm(
                torch.flatten(grad_f_scaled, start_dim=1), p=2, dim=1)
            grad_norm_scaled = grad_norm_scaled.view(-1, 1)
            f_hat_scaled = f_scaled / (grad_norm_scaled + torch.abs(f_scaled))

            print(f'{step:2d} '
                  f'{f_hat_scaled.item():+11.7f}, '
                  f'{f_scaled.item():+11.7f}, '
                  f'{grad_norm_scaled.item():+11.7f}, '
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
