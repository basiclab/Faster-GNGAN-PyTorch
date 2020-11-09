import os
import json

import torch
import torch.optim as optim
from absl import flags, app
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from models import gn_gan, sn_gan
from common.losses import HingeLoss
from common.datasets import get_dataset
from common.score.score import get_inception_and_fid_score
from common.utils import (
    ema, images_generator, save_images, module_no_grad, infiniteloop, set_seed)


net_G_models = {
    'gn-res128': gn_gan.ResGenerator128,
    'sn-res128': sn_gan.ResGenerator128,
}

net_D_models = {
    'gn-res128': gn_gan.ResDiscriminator128,
    'sn-res128': sn_gan.ResDiscriminator128,
}

net_GD_models = {
    'gn-res128': gn_gan.GenDis,
    'sn-res128': sn_gan.GenDis,
}

datasets = [
    'imagenet128', 'imagenet128.hdf5',
    'celebhq128', 'celebhq128.hdf5',
    'lsun_church_outdoor', 'lsun_church_outdoor.hdf5',]


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'celebhq128', datasets, "select dataset")
flags.DEFINE_enum('arch', 'gn-res128', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 200000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 200000, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 16, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 5000, "start step for ema")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GN-GAN_CELEBHQ128_RES_0', 'log folder')
flags.DEFINE_string('fid_cache', './stats/celebhq_val128.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_integer('num_images', 10000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(
        torch.load(os.path.join(FLAGS.logdir, 'model.pt'))['ema_G'])

    images = images_generator(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.G_batch_size)
    save_images(
        images, os.path.join(FLAGS.logdir, 'generate'), verbose=True)


def evaluate(net_G):
    images = images_generator(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.batch_size)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.eval_use_torch, parallel=FLAGS.parallel, verbose=True)
    del images
    return (IS, IS_std), FID


def train():
    dataset = get_dataset(FLAGS.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    # model
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    ema_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    if FLAGS.arch.startswith('gn'):
        net_D = gn_gan.GradNorm(net_D)
    net_GD = net_GD_models[FLAGS.arch](net_G, net_D)

    if FLAGS.parallel:
        net_GD = torch.nn.DataParallel(net_GD)

    # ema
    ema(net_G, ema_G, decay=0)

    # loss
    loss_fn = HingeLoss()

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
        start = ckpt['step'] + 1
        writer = SummaryWriter(FLAGS.logdir)
        writer_ema = SummaryWriter(FLAGS.logdir + "_ema")
        del ckpt
    else:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        writer = SummaryWriter(FLAGS.logdir)
        writer_ema = SummaryWriter(FLAGS.logdir + "_ema")
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_z = torch.split(fixed_z, FLAGS.batch_size, dim=0)
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
            f.write(FLAGS.flags_into_string())
        writer.add_text(
            "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))
        real = []
        for x, _ in dataloader:
            real.append(x)
            if len(real) * FLAGS.batch_size >= FLAGS.sample_size:
                real = torch.cat(real, dim=0)[:FLAGS.sample_size]
                break
        grid = (make_grid(real) + 1) / 2
        writer.add_image('real_sample', grid)
        writer.flush()
        start = 1

    D_size = 0
    for param in net_D.parameters():
        D_size += param.data.nelement()
    G_size = 0
    for param in net_G.parameters():
        G_size += param.data.nelement()
    print('D params: %d, G params: %d' % (D_size, G_size))

    z = torch.randn(2 * FLAGS.batch_size, FLAGS.z_dim, requires_grad=False)
    z = z.to(device)

    looper = infiniteloop(dataloader)
    with trange(start, FLAGS.total_steps + 1, dynamic_ncols=True,
                initial=start - 1, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0

            # Discriminator
            for _ in range(FLAGS.n_dis):
                optim_D.zero_grad()
                for _ in range(FLAGS.D_accumulation):
                    real, _ = next(looper)
                    real = real.to(device)
                    z.normal_()
                    pred_real, pred_fake = net_GD(z[: FLAGS.batch_size], real)
                    loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                    loss = loss / float(FLAGS.D_accumulation)
                    loss.backward()

                    loss_sum += loss.cpu().item()
                    loss_real_sum += loss_real.cpu().item()
                    loss_fake_sum += loss_fake.cpu().item()
                optim_D.step()

            loss = loss_sum / FLAGS.n_dis / FLAGS.D_accumulation
            loss_real = loss_real_sum / FLAGS.n_dis / FLAGS.D_accumulation
            loss_fake = loss_fake_sum / FLAGS.n_dis / FLAGS.D_accumulation

            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_real', loss_real, step)
            writer.add_scalar('loss_fake', loss_fake, step)

            pbar.set_postfix(
                loss_real='%.3f' % loss_real,
                loss_fake='%.3f' % loss_fake)

            # Generator
            net_G.train()
            optim_G.zero_grad()
            for _ in range(FLAGS.G_accumulation):
                z.normal_()
                with module_no_grad(net_D):
                    loss = loss_fn(net_GD(z))
                loss = loss / float(FLAGS.G_accumulation)
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

            # sample from fixed z
            if step == 1 or step % FLAGS.sample_step == 0:
                fake_imgs = []
                with torch.no_grad():
                    for fixed_z_batch in fixed_z:
                        fake = ema_G(fixed_z_batch).cpu()
                        fake_imgs.append((fake + 1) / 2)
                    grid = make_grid(torch.cat(fake_imgs, dim=0))
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            # evaluate IS, FID and save model
            if step == 1 or step % FLAGS.eval_step == 0:
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
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.parallel:
                    eval_net_G = torch.nn.DataParallel(net_G)
                    eval_ema_G = torch.nn.DataParallel(ema_G)
                else:
                    eval_net_G = net_G
                    eval_ema_G = ema_G
                (net_G_IS, net_G_IS_std), net_G_FID = evaluate(eval_net_G)
                (ema_G_IS, ema_G_IS_std), ema_G_FID = evaluate(eval_ema_G)
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
