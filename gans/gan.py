import os
import json
from copy import deepcopy

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from models import gn_gan, sn_gan
from common.losses import loss_fns
from common.dataset import get_dataset
from common.score.score import get_inception_and_fid_score
from common.utils import (
    generate_images, save_images, module_no_grad, infiniteloop, set_seed)

FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'stl10', 'imagenet128', 'imagenet128.hdf5'], "select dataset")
flags.DEFINE_enum('arch', 'res32', gn_gan.generators.keys(), "architecture")
flags.DEFINE_enum('norm', 'GN', ['GN', 'SN'], "normalization techniques")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 0, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_float('cr', 0, "weight for consistency regularization")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GN-GAN_CIFAR10_RES_0', 'log folder')
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 10000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    if FLAGS.norm == 'GN':
        Generator = gn_gan.generators[FLAGS.arch]
        net_G = Generator(FLAGS.z_dim).to(device)
    if FLAGS.norm == 'SN':
        Generator = sn_gan.generators[FLAGS.arch]
        net_G = Generator(FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])

    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.batch_size,
        verbose=True)
    save_images(images=images, output_dir=FLAGS.output_dir)


def evaluate(net_G):
    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.batch_size,
        verbose=False)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, use_torch=FLAGS.eval_use_torch,
        parallel=FLAGS.parallel)
    del images
    return (IS, IS_std), FID


def consistency_loss(net_D, real, pred_real):
    consistency_transforms = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.ToPILImage(mode='RGB'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    aug_real = deepcopy(real.cpu())
    for idx, img in enumerate(aug_real):
        aug_real[idx] = consistency_transforms(img)
    aug_real = aug_real.to(device)
    loss = ((net_D(aug_real) - pred_real) ** 2).mean()
    return loss


def train():
    dataset = get_dataset(FLAGS.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    # model
    if FLAGS.norm == 'GN':
        Generator = gn_gan.generators[FLAGS.arch]
        Discriminator = gn_gan.discriminators[FLAGS.arch]
        net_G = Generator(FLAGS.z_dim).to(device)
        net_D = Discriminator().to(device)
        net_D = gn_gan.GradNorm(net_D)
        net_GD = gn_gan.GenDis(net_G, net_D)
    if FLAGS.norm == 'SN':
        Generator = sn_gan.generators[FLAGS.arch]
        Discriminator = sn_gan.discriminators[FLAGS.arch]
        net_G = Generator(FLAGS.z_dim).to(device)
        net_D = Discriminator().to(device)
        net_GD = sn_gan.GenDis(net_G, net_D)

    if FLAGS.parallel:
        net_GD = torch.nn.DataParallel(net_GD)

    # loss
    loss_fn = loss_fns[FLAGS.loss]()

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
        fixed_z = ckpt['fixed_z']
        start = ckpt['step'] + 1
        writer = SummaryWriter(FLAGS.logdir)
        del ckpt
    else:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        writer = SummaryWriter(FLAGS.logdir)
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

    z = torch.randn(2 * FLAGS.batch_size, FLAGS.z_dim, requires_grad=False)
    z = z.to(device)

    looper = infiniteloop(dataloader)
    with trange(start, FLAGS.total_steps + 1, dynamic_ncols=True,
                initial=start, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            loss_list = []
            loss_real_list = []
            loss_fake_list = []

            if FLAGS.cr > 0:
                loss_cr_list = []

            # Discriminator
            net_D.train()
            for _ in range(FLAGS.n_dis):
                optim_D.zero_grad()
                real, _ = next(looper)
                real = real.to(device)
                z.normal_()
                pred_real, pred_fake = net_GD(z[: FLAGS.batch_size], real)
                loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                loss_all = loss
                if FLAGS.cr > 0:
                    loss_cr = consistency_loss(net_D, real, pred_real)
                    loss_all += FLAGS.cr * loss_cr
                    loss_cr_list.append(loss_cr.detach())
                loss_all.backward()
                loss_list.append(loss.detach())
                loss_real_list.append(loss_real.detach())
                loss_fake_list.append(loss_fake.detach())
                optim_D.step()

            loss = torch.mean(torch.stack(loss_list))
            loss_real = torch.mean(torch.stack(loss_real_list))
            loss_fake = torch.mean(torch.stack(loss_fake_list))
            if FLAGS.cr > 0:
                loss_cr = torch.mean(torch.stack(loss_cr_list))
            if FLAGS.loss == 'was':
                loss = -loss
            pbar.set_postfix(
                loss='%.4f' % loss,
                loss_real='%.4f' % loss_real,
                loss_fake='%.4f' % loss_fake)

            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('loss_real', loss_real.item(), step)
            writer.add_scalar('loss_fake', loss_fake.item(), step)
            if FLAGS.cr > 0:
                writer.add_scalar('loss_cr', loss_cr.item(), step)

            # Generator
            net_G.train()
            optim_G.zero_grad()
            z.normal_()
            with module_no_grad(net_D):
                loss = loss_fn(net_GD(z))
            loss.backward()
            optim_G.step()

            # scheduler
            sched_G.step()
            sched_D.step()

            if step == 1 or step % FLAGS.sample_step == 0:
                fake_imgs = []
                with torch.no_grad():
                    for fixed_z_batch in fixed_z:
                        fake = net_G(fixed_z_batch).cpu()
                        fake_imgs.append((fake + 1) / 2)
                    grid = make_grid(torch.cat(fake_imgs, dim=0))
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                ckpt = {
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'step': step,
                    'fixed_z': fixed_z,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.parallel:
                    eval_net_G = torch.nn.DataParallel(net_G)
                else:
                    eval_net_G = net_G
                (net_G_IS, net_G_IS_std), net_G_FID = evaluate(eval_net_G)
                pbar.write(
                    "%6d/%6d "
                    "IS: %6.3f(%.3f), FID: %7.3f" % (
                        step, FLAGS.total_steps,
                        net_G_IS, net_G_IS_std, net_G_FID))
                writer.add_scalar('IS', net_G_IS, step)
                writer.add_scalar('IS_std', net_G_IS_std, step)
                writer.add_scalar('FID', net_G_FID, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    f.write(json.dumps(
                        {
                            'step': step,
                            'IS': net_G_IS,
                            'IS_std': net_G_IS_std,
                            'FID': net_G_FID,
                        }) + "\n"
                    )
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
