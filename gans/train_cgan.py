import os
import json
import math

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from models import sn_cgan, gn_cgan, gn_biggan, sn_biggan, gn_gan
from common.losses import HingeLoss
from common.datasets import get_dataset
from common.score.score import get_inception_and_fid_score
from common.utils import (
    ema, generate_images, save_images, module_no_grad, infiniteloop, set_seed,
    record_weight_norm)


net_G_models = {
    'gn-res32': gn_cgan.ResGenerator32,
    'gn-biggan32': gn_biggan.Generator32,
    'sn-res32': sn_cgan.ResGenerator32,
    'sn-biggan32': sn_biggan.Generator32,
}

net_D_models = {
    'gn-res32': gn_cgan.ResDiscriminator32,
    'gn-biggan32': gn_biggan.Discriminator32,
    'sn-res32': sn_cgan.ResDiscriminator32,
    'sn-biggan32': sn_biggan.Discriminator32,
}

net_GD_models = {
    'gn-res32': gn_cgan.GenDis,
    'gn-biggan32': gn_biggan.GenDis,
    'sn-res32': sn_cgan.GenDis,
    'sn-biggan32': sn_biggan.GenDis,
}


datasets = ['cifar10']


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'cifar10', datasets, "dataset")
flags.DEFINE_enum('arch', 'gn-biggan32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('ch', 64, 'base channel size of BigGAN')
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('total_steps', 125000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 0, 'apply linearly decay to lr')
flags.DEFINE_integer('G_batch_size', 50, "batch size")
flags.DEFINE_integer('D_batch_size', 50, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('G_lr', 1e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.999], "for Adam")
flags.DEFINE_integer('n_dis', 4, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('scale', 1., "boundary value of hinge loss")
flags.DEFINE_float('cr', 0, "weight for consistency regularization")
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GN-BIGGAN_CIFAR10_0', 'log folder')
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
# generate sample
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    net_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    net_G = net_G.to(device)
    if os.path.isfile(os.path.join(FLAGS.logdir, 'best_model.pt')):
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'best_model.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'model.pt'))

    net_G.load_state_dict(ckpt['ema_G'])

    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        n_classes=FLAGS.n_classes,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.G_batch_size,
        verbose=True)
    save_images(
        images, os.path.join(FLAGS.logdir, 'generate'), verbose=True)


def evaluate(net_G):
    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        n_classes=FLAGS.n_classes,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.G_batch_size,
        verbose=True)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.eval_use_torch, verbose=True)
    del images
    return (IS, IS_std), FID


def consistency_loss(net_D, x_real, y_real, pred_real):
    consistency_transforms = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.ToPILImage(mode='RGB'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    aug_real = x_real.detach().clone().cpu()
    for idx, img in enumerate(aug_real):
        aug_real[idx] = consistency_transforms(img)
    aug_real = aug_real.to(device)
    loss = ((net_D(aug_real, y=y_real) - pred_real) ** 2).mean()
    return loss


def train():
    dataset = get_dataset(FLAGS.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.D_batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    # model
    net_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    net_G = net_G.to(device)
    ema_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    ema_G = ema_G.to(device)
    net_D = net_D_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes).to(device)
    if FLAGS.arch.startswith('gn'):
        net_D = gn_gan.GradNorm(net_D)
    net_GD = net_GD_models[FLAGS.arch](net_G, net_D)

    # ema
    ema(net_G, ema_G, decay=0)

    # loss
    loss_fn = HingeLoss(FLAGS.scale)

    # optimizer
    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    # scheduler
    def decay_rate(step):
        return 1 - max(step - FLAGS.lr_decay_start, 0) / FLAGS.total_steps
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=decay_rate)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=decay_rate)

    if FLAGS.resume:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'model.pt'))
        net_G.load_state_dict(ckpt['net_G'])
        net_D.load_state_dict(ckpt['net_D'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        sched_G.load_state_dict(ckpt['sched_G'])
        sched_D.load_state_dict(ckpt['sched_D'])
        ema_G.load_state_dict(ckpt['ema_G'])
        fixed_z = ckpt['fixed_z']
        fixed_y = ckpt['fixed_y']
        start = ckpt['step'] + 1
        if 'best_ID' in ckpt and 'best_FID' in ckpt:
            best_IS, best_FID = ckpt['best_ID'], ckpt['best_FID']
        else:
            best_IS, best_FID = 0, 999
        writer = SummaryWriter(FLAGS.logdir)
        writer_ema = SummaryWriter(FLAGS.logdir + "_ema")
        del ckpt
    else:
        # sample fixed z and y
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        writer = SummaryWriter(FLAGS.logdir)
        writer_ema = SummaryWriter(FLAGS.logdir + "_ema")
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_z = torch.split(fixed_z, FLAGS.G_batch_size, dim=0)
        fixed_y = torch.randint(
            FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
        fixed_y = torch.split(fixed_y, FLAGS.G_batch_size, dim=0)
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
            f.write(FLAGS.flags_into_string())
        writer.add_text(
            "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

        # sample real data
        real = []
        for x, _ in dataloader:
            real.append(x)
            if len(real) * x.shape[0] >= FLAGS.sample_size:
                real = torch.cat(real, dim=0)[:FLAGS.sample_size]
                break
        grid = (make_grid(real) + 1) / 2
        writer.add_image('real_sample', grid)
        writer.flush()
        start = 1
        best_IS, best_FID = 0, 999

    looper = infiniteloop(dataloader)
    with trange(start, FLAGS.total_steps + 1, dynamic_ncols=True,
                initial=start - 1, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0
            loss_cr_sum = 0
            net_D.train()
            net_G.train()

            # Discriminator
            for _ in range(FLAGS.n_dis):
                x_real, y_real = next(looper)
                x_real, y_real = x_real.to(device), y_real.to(device)
                z = torch.randn(
                    FLAGS.D_batch_size, FLAGS.z_dim, device=device)
                y = torch.randint(
                    FLAGS.n_classes, (FLAGS.D_batch_size,), device=device)
                pred_real, pred_fake = net_GD(z, y, x_real, y_real)
                loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                if FLAGS.cr > 0:
                    loss_cr = consistency_loss(
                        net_D, x_real, y_real, pred_real)
                else:
                    loss_cr = torch.tensor(0.)
                loss_all = loss + FLAGS.cr * loss_cr

                optim_D.zero_grad()
                loss_all.backward()
                optim_D.step()

                loss_sum += loss.cpu().item()
                loss_real_sum += loss_real.cpu().item()
                loss_fake_sum += loss_fake.cpu().item()
                loss_cr_sum += loss_cr.cpu().item()

            loss = loss_sum / FLAGS.n_dis
            loss_real = loss_real_sum / FLAGS.n_dis
            loss_fake = loss_fake_sum / FLAGS.n_dis
            loss_cr = loss_cr_sum / FLAGS.n_dis

            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_real', loss_real, step)
            writer.add_scalar('loss_fake', loss_fake, step)
            writer.add_scalar('loss_cr', loss_cr, step)

            pbar.set_postfix(
                loss_real='%.3f' % loss_real,
                loss_fake='%.3f' % loss_fake)

            # Generator
            z = torch.randn(
                FLAGS.G_batch_size, FLAGS.z_dim, device=device)
            y = torch.randint(
                FLAGS.n_classes, (FLAGS.G_batch_size,), device=device)
            with module_no_grad(net_D):
                loss = loss_fn(net_GD(z, y))
            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            # scheduler
            sched_G.step()
            sched_D.step()

            # ema
            if step < FLAGS.ema_start:
                decay = 0
            else:
                decay = FLAGS.ema_decay
            ema(net_G, ema_G, decay)

            if step == 1 or step % FLAGS.sample_step == 0:
                fake_imgs = []
                with torch.no_grad():
                    for fixed_z_batch, fixed_y_batch in zip(fixed_z, fixed_y):
                        fake = ema_G(fixed_z_batch, fixed_y_batch).cpu()
                        fake_imgs.append((fake + 1) / 2)
                    grid = make_grid(torch.cat(fake_imgs, dim=0))
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                record_weight_norm(net_G, net_D, writer, step)
                (net_G_IS, net_G_IS_std), net_G_FID = evaluate(net_G)
                (ema_G_IS, ema_G_IS_std), ema_G_FID = evaluate(ema_G)
                if not math.isnan(ema_G_FID):
                    save_as_best = (ema_G_FID < best_FID)
                elif not math.isnan(ema_G_IS):
                    save_as_best = (ema_G_IS > best_IS)
                else:
                    save_as_best = False
                if save_as_best:
                    best_FID = best_FID if math.isnan(ema_G_FID) else ema_G_FID
                    best_IS = best_IS if math.isnan(ema_G_IS) else ema_G_IS
                ckpt = {
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'ema_G': ema_G.state_dict(),
                    'step': step,
                    'fixed_z': fixed_z,
                    'fixed_y': fixed_y,
                    'best_IS': best_IS,
                    'best_FID': best_FID,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if save_as_best:
                    torch.save(
                        ckpt, os.path.join(FLAGS.logdir, 'best_model.pt'))
                pbar.write(
                    "%6d/%6d "
                    "IS:%6.3f(%.3f), FID:%7.3f, "
                    "IS(EMA):%6.3f(%.3f), FID(EMA):%7.3f" % (
                        step, FLAGS.total_steps,
                        net_G_IS, net_G_IS_std, net_G_FID,
                        ema_G_IS, ema_G_IS_std, ema_G_FID))
                writer.add_scalar('IS', net_G_IS, step)
                writer.add_scalar('IS_std', net_G_IS_std, step)
                writer.add_scalar('FID', net_G_FID, step)
                writer_ema.add_scalar('IS', ema_G_IS, step)
                writer_ema.add_scalar('IS_std', ema_G_IS_std, step)
                writer_ema.add_scalar('FID', ema_G_FID, step)
                writer.flush()
                writer_ema.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    f.write(json.dumps(
                        {
                            'step': step,
                            'IS': net_G_IS,
                            'IS_std': net_G_IS_std,
                            'FID': net_G_FID,
                            'IS(EMA)': ema_G_IS,
                            'IS_std(EMA)': ema_G_IS_std,
                            'FID(EMA)': ema_G_FID
                        }) + "\n"
                    )
    writer.close()
    writer_ema.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
