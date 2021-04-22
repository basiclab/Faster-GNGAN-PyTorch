import torch
import torch.utils.benchmark as benchmark
import torchvision.transforms as transforms

from source.models import sn_gan, gp_gan, gn_gan
from source.models.gn_gan import apply_grad_norm_hook


device = torch.device('cuda:0')
consistency_transforms = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1) / 2),
    transforms.ToPILImage(mode='RGB'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def consistency_loss(net_D, real, pred_real):
    aug_real = real.detach().clone().cpu()
    for idx, img in enumerate(aug_real):
        aug_real[idx] = consistency_transforms(img)
    aug_real = aug_real.to(device)
    loss = ((net_D(aug_real) - pred_real) ** 2).mean()
    return loss


def gradient_penalty(net_D, real, fake, center=1):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - center) ** 2)
    return loss_gp


def train_D_step(G, D, batch_size, z_dim, gp=0, cr=0):
    torch.cuda.synchronize()
    z = torch.randn((batch_size, z_dim), device=device)
    real = torch.randn((batch_size, 3, 32, 32), device=device)
    with torch.no_grad():
        fake = G(z)
    pred = D(torch.cat([real, fake], dim=0))
    pred_real, pred_fake = torch.split(pred, [batch_size, batch_size])
    loss = pred_fake.mean() - pred_real.mean()
    if gp > 0:
        loss = loss + gp * gradient_penalty(D, real, fake)
    if cr > 0:
        loss = loss + cr * consistency_loss(D, real, pred_real)
    loss.backward()
    torch.cuda.synchronize()


def train_G_step(G, D, batch_size, z_dim):
    torch.cuda.synchronize()
    z = torch.randn((batch_size * 2, z_dim), device=device)
    loss = -D(G(z)).mean()
    loss.backward()
    torch.cuda.synchronize()


def train_GN_G_step(G, D, batch_size, z_dim):
    torch.cuda.synchronize()
    z = torch.randn((batch_size * 2, z_dim), device=device)
    x = G(z)
    y = D.forward_impl(x)
    apply_grad_norm_hook(x, y)
    loss = -y.mean()
    loss.backward()
    torch.cuda.synchronize()


if __name__ == '__main__':
    z_dim = 128
    batch_size = 64

    results = []

    print('GAN...')
    G = gp_gan.ResGenerator32(z_dim).to(device)
    D = gp_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'GAN',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    print('SNGAN...')
    G = sn_gan.ResGenerator32(z_dim).to(device)
    D = sn_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'SNGAN',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    print('CRSNGAN...')
    G = sn_gan.ResGenerator32(z_dim).to(device)
    D = sn_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim, cr=cr)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim, 'cr': 10}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'CRSNGAN',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    print('WGANGP...')
    G = gp_gan.ResGenerator32(z_dim).to(device)
    D = gp_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim, gp)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim, 'gp': 10}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'WGANGP',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    print('GNGAN...')
    G = gn_gan.ResGenerator32(z_dim).to(device)
    D = gn_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'GNGAN',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    print('GNGAN fast...')
    G = gn_gan.ResGenerator32(z_dim).to(device)
    D = gn_gan.ResDiscriminator32().to(device)
    d = benchmark.Timer(
        stmt='train_D_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_D_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    g = benchmark.Timer(
        stmt='train_GN_G_step(G, D, batch_size, z_dim)',
        setup='from __main__ import train_GN_G_step',
        globals={
            'G': G, 'D': D, 'batch_size': batch_size, 'z_dim': z_dim}
        ).blocked_autorange(min_run_time=5)
    results.append({
        'Method': 'GNGAN(fast)',
        'D(it/ms)': d.median,
        'G(it/ms)': g.median,
    })

    formats = {
        'Method': '%s',
        'D(it/ms)': '%.2f',
        'G(it/ms)': '%.2f',
    }
    for result in results:
        for k, v in result.items():
            if 'f' in formats[k]:
                v = v * 1000
            result[k] = formats[k] % v
    lens = dict()
    for name in formats.keys():
        length = max([len(result[name]) for result in results])
        length = max(length, len(name))
        lens[name] = length

    head = (
        "|" +
        "|".join(['%-*s' % (lens[name], name) for name in formats.keys()]) +
        "|")
    print(head)
    sep = (
        "|" +
        "|".join([(lens[name] * '-') for name in formats.keys()]) +
        "|")
    print(sep)
    for result in results:
        line = "|".join(
            ['%-*s' % (lens[name], v) for name, v in result.items()])
        line = "|" + line + "|"
        print(line)
