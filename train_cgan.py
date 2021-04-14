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

from source.models import gn_biggan
from source.losses import HingeLoss
from source.datasets import get_dataset
from source.utils import (
    ema, save_images, module_no_grad, infiniteloop, set_seed)
from metrics.score.both import get_inception_score_and_fid


net_G_models = {
    'gn-biggan32': gn_biggan.Generator32,
}

net_D_models = {
    'gn-biggan32': gn_biggan.Discriminator32,
}


datasets = ['cifar10.32.raw']


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'cifar10.32.raw', datasets, "dataset")
flags.DEFINE_enum('arch', 'gn-biggan32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('ch', 64, 'base channel size of BigGAN')
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('total_steps', 125000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 125000, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 50, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('G_lr', 1e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.999], "for Adam")
flags.DEFINE_integer('n_dis', 4, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('cr', 0, "weight for consistency regularization")
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('save_step', 20000, "save model every this step")
flags.DEFINE_integer('num_images', 50000, '# images for evaluation')
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GN-BIGGAN_CIFAR10_0', 'log folder')
flags.DEFINE_string('fid_stats', './stats/cifar10.train.npz', 'FID cache')
# generate sample
flags.DEFINE_bool('generate', False, 'generate images from pretrained model')
flags.DEFINE_string('output', None, 'path to output directory')


device = torch.device('cuda:0')


def generate_images(net_G):
    images = []
    with torch.no_grad():
        for _ in trange(0, FLAGS.num_images, FLAGS.batch_size,
                        ncols=0, leave=False):
            z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
            y = torch.randint(FLAGS.n_classes, (FLAGS.batch_size,)).to(device)
            fake = (net_G(z, y) + 1) / 2
            images.append(fake.cpu())
    images = torch.cat(images, dim=0)
    return images[:FLAGS.num_images]


def generate():
    net_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    net_G = net_G.to(device)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'best_model.pt'))
    net_G.load_state_dict(ckpt['ema_G'])

    images = generate_images(net_G=net_G)
    if FLAGS.output is not None:
        save_path = FLAGS.output
    else:
        save_path = os.path.join(FLAGS.logdir, 'output')
    save_images(images, save_path, verbose=True)
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, FLAGS.fid_stats, use_torch=FLAGS.eval_use_torch, verbose=True)
    print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))


def evaluate(net_G):
    images = generate_images(net_G=net_G)
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, FLAGS.fid_stats, use_torch=FLAGS.eval_use_torch, verbose=True)
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
        dataset=dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_dis,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True)
    looper = infiniteloop(dataloader)

    # model
    net_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    net_G = net_G.to(device)
    ema_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    ema_G = ema_G.to(device)
    net_D = net_D_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes).to(device)
    net_GD = gn_biggan.GenDis(net_G, net_D)

    # ema
    ema(net_G, ema_G, decay=0)

    # loss
    loss_fn = HingeLoss()

    # optimizer
    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    # scheduler
    def decay_rate(step):
        period = max(FLAGS.total_steps - FLAGS.lr_decay_start, 1)
        return 1 - max(step - FLAGS.lr_decay_start, 0) / period
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=decay_rate)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=decay_rate)

    D_size = 0
    for param in net_D.parameters():
        D_size += param.data.nelement()
    G_size = 0
    for param in net_G.parameters():
        G_size += param.data.nelement()
    print('D params: %d, G params: %d' % (D_size, G_size))

    writer = SummaryWriter(FLAGS.logdir)
    if FLAGS.resume:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'model.pt'))
        net_G.load_state_dict(ckpt['net_G'])
        net_D.load_state_dict(ckpt['net_D'])
        ema_G.load_state_dict(ckpt['ema_G'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        sched_G.load_state_dict(ckpt['sched_G'])
        sched_D.load_state_dict(ckpt['sched_D'])
        fixed_z = ckpt['fixed_z']
        fixed_y = ckpt['fixed_y']
        # start value
        start = ckpt['step'] + 1
        best_IS, best_FID = ckpt['best_IS'], ckpt['best_FID']
        del ckpt
    else:
        # sample fixed z and y
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_y = torch.randint(
            FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
        # start value
        start, best_IS, best_FID = 1, 0, 999

        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
            f.write(FLAGS.flags_into_string())
        real = next(iter(dataloader))[0][:FLAGS.sample_size]
        writer.add_image('real_sample', make_grid((real + 1) / 2))
        writer.flush()

    with trange(start, FLAGS.total_steps + 1, ncols=0,
                initial=start - 1, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0
            loss_cr_sum = 0

            x, y = next(looper)
            x = iter(torch.split(x, FLAGS.batch_size))
            y = iter(torch.split(y, FLAGS.batch_size))
            # Discriminator
            for _ in range(FLAGS.n_dis):
                optim_D.zero_grad()
                x_real, y_real = next(x).to(device), next(y).to(device)
                z_ = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                y_ = torch.randint(
                    FLAGS.n_classes, (FLAGS.batch_size,)).to(device)
                pred_real, pred_fake = net_GD(z_, y_, x_real, y_real)
                loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                if FLAGS.cr > 0:
                    loss_cr = consistency_loss(
                        net_D, x_real, y_real, pred_real)
                else:
                    loss_cr = torch.tensor(0.)
                loss_all = loss + FLAGS.cr * loss_cr
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
            optim_G.zero_grad()
            with module_no_grad(net_D):
                z_ = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                y_ = torch.randint(
                    FLAGS.n_classes, (FLAGS.batch_size,)).to(device)
                loss = loss_fn(net_GD(z_, y_))
                loss.backward()
            optim_G.step()

            # ema
            if step < FLAGS.ema_start:
                decay = 0
            else:
                decay = FLAGS.ema_decay
            ema(net_G, ema_G, decay)

            # scheduler
            sched_G.step()
            sched_D.step()

            # sample from fixed z and y
            if step == 1 or step % FLAGS.sample_step == 0:
                with torch.no_grad():
                    fake_net = net_G(fixed_z, fixed_y).cpu()
                    fake_ema = ema_G(fixed_z, fixed_y).cpu()
                grid_net = (make_grid(fake_net) + 1) / 2
                grid_ema = (make_grid(fake_ema) + 1) / 2
                writer.add_image('sample_ema', grid_ema, step)
                writer.add_image('sample', grid_net, step)
                save_image(
                    grid_ema,
                    os.path.join(FLAGS.logdir, 'sample', '%d.png' % step))

            # evaluate IS, FID and save model
            if step == 1 or step % FLAGS.eval_step == 0:
                (net_IS, net_IS_std), net_FID = evaluate(net_G)
                (ema_IS, ema_IS_std), ema_FID = evaluate(ema_G)
                if not math.isnan(ema_FID) and not math.isnan(best_FID):
                    save_as_best = (ema_FID < best_FID)
                else:
                    save_as_best = (ema_IS > best_IS)
                if save_as_best:
                    best_IS = ema_IS
                    best_FID = best_FID
                ckpt = {
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'ema_G': ema_G.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'fixed_z': fixed_z,
                    'fixed_y': fixed_y,
                    'best_IS': best_IS,
                    'best_FID': best_FID,
                    'step': step,
                }
                if step == 1 or step % FLAGS.save_step == 0:
                    torch.save(
                        ckpt, os.path.join(FLAGS.logdir, '%06d.pt' % step))
                if save_as_best:
                    torch.save(
                        ckpt, os.path.join(FLAGS.logdir, 'best_model.pt'))
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                metrics = {
                    'IS': net_IS,
                    'IS_std': net_IS_std,
                    'FID': net_FID,
                    'IS_EMA': ema_IS,
                    'IS_std_EMA': ema_IS_std,
                    'FID_EMA': ema_FID,
                }
                pbar.write(
                    "{}/{} ".format(step, FLAGS.total_steps) +
                    "IS: {IS:6.3f}({IS_std:.3f}), FID: {FID:.3f}, "
                    "IS_EMA: {IS_EMA:6.3f}({IS_std_EMA:.3f}), "
                    "FID_EMA: {FID_EMA:.3f}, ".format(**metrics))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
