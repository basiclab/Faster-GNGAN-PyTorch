import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

import models.gngan as gngan
import models.biggan as biggan
import common.losses as losses
from common.utils import (
    generate_conditional_imgs, infiniteloop, set_seed, module_require_grad)
from common.score.score import get_inception_and_fid_score


net_G_models = {
    'biggan32': biggan.Generator32,
    'biggan128': biggan.Generator128,
}

net_D_models = {
    'biggan32': biggan.Discriminator32,
    'biggan128': biggan.Discriminator128,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}


FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'imagenet'], "dataset")
flags.DEFINE_enum('arch', 'biggan32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('ch', 64, 'base channel size of BigGAN')
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_bool('GN', False, "enable gradient penalty")
flags.DEFINE_bool('scheduler', True, 'apply linear learing rate decay')
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GNGAN_CIFAR10_BIGGAN', 'log folder')
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
flags.DEFINE_bool('record', True, "record inception score and FID score")
# flags.DEFINE_integer('max_ckpts', 3, 'the number of recent checkpoints kept')

# generate sample
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])
    net_G.eval()

    counter = 0
    os.makedirs(FLAGS.output)
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            y = torch.randint(FLAGS.n_classes, size=(FLAGS.sample_size,))
            x = net_G(z, y).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1


def train():
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'imagenet':
        dataset = datasets.ImageFolder(
            './data/ILSVRC2012/train',
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    net_G = net_G_models[FLAGS.arch](
        FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim).to(device)
    if FLAGS.GN:
        # if enable GN, disable all spectral norm in discriminator
        net_D = net_D_models[FLAGS.arch](
            FLAGS.ch, FLAGS.n_classes, sn=(lambda x: x)).to(device)
        net_D = gngan.GradNorm(net_D)
    else:
        net_D = net_D_models[FLAGS.arch](
            FLAGS.ch, FLAGS.n_classes).to(device)
    net_D_G = biggan.DisGen(net_D, net_G)

    if FLAGS.parallel:
        net_D_G = torch.nn.DataParallel(net_D_G)
    loss_fn = loss_fns[FLAGS.loss]()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    if FLAGS.scheduler:
        sched_G = optim.lr_scheduler.LambdaLR(
            optim_G, lambda step: 1 - step / FLAGS.total_steps)
        sched_D = optim.lr_scheduler.LambdaLR(
            optim_D, lambda step: 1 - step / FLAGS.total_steps)

    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    sample_y = torch.randint(
        FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(FLAGS.n_dis):
                loss_list = []
                loss_real_list = []
                loss_fake_list = []
                optim_D.zero_grad()
                for __ in range(FLAGS.D_accumulation):
                    real, y_real = next(looper)
                    real, y_real = real.to(device), y_real.to(device)
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    y = torch.randint(
                        FLAGS.n_classes, size=(FLAGS.batch_size,)).to(device)
                    net_D_real, net_D_fake = net_D_G(z, y, real, y_real)
                    loss, loss_real, loss_fake = loss_fn(
                        net_D_real, net_D_fake)
                    loss_list.append(loss.detach())
                    loss_real_list.append(loss_real.detach())
                    loss_fake_list.append(loss_fake.detach())
                    loss = loss / float(FLAGS.D_accumulation)
                    loss.backward()
                optim_D.step()
                loss = torch.mean(torch.stack(loss_list))
                loss_real = torch.mean(torch.stack(loss_real_list))
                loss_fake = torch.mean(torch.stack(loss_fake_list))

                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)

            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('loss_real', loss_real.item(), step)
            writer.add_scalar('loss_fake', loss_fake.item(), step)

            # Generator
            with module_require_grad(net_D, False):
                optim_G.zero_grad()
                for _ in range(FLAGS.G_accumulation):
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    y = torch.randint(
                        FLAGS.n_classes, size=(FLAGS.batch_size,)).to(device)
                    loss = loss_fn(net_D_G(z, y))
                    loss = loss / float(FLAGS.G_accumulation)
                    loss.backward()
                optim_G.step()

            # scheduler
            if FLAGS.scheduler:
                sched_G.step()
                sched_D.step()

            # update progress & log something periodically
            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                # TODO: bachwise
                with torch.no_grad():
                    fake = net_G(sample_z, sample_y).cpu()
                    grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step % FLAGS.eval_step == 0:
                ckpt = {
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                }
                if FLAGS.parallel:
                    ckpt.update({
                        'net_G': net_G.module.state_dict(),
                        'net_D': net_D.module.state_dict(),
                    })
                else:
                    ckpt.update({
                        'net_G': net_G.state_dict(),
                        'net_D': net_D.state_dict(),
                    })
                if FLAGS.scheduler:
                    ckpt.update({
                        'sched_G': sched_G.state_dict(),
                        'sched_D': sched_D.state_dict(),
                    })
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_conditional_imgs(
                        net_G, device, FLAGS.n_classes,
                        FLAGS.z_dim, FLAGS.num_images, FLAGS.batch_size)
                    is_score, fid_score = get_inception_and_fid_score(
                        imgs, device, FLAGS.fid_cache, verbose=True)
                    pbar.write(
                        "%s/%s Inception Score: %.3f(%.5f), "
                        "FID Score: %6.3f" % (
                            step, FLAGS.total_steps, is_score[0], is_score[1],
                            fid_score))
                    writer.add_scalar('inception_score', is_score[0], step)
                    writer.add_scalar('inception_score_std', is_score[1], step)
                    writer.add_scalar('fid_score', fid_score, step)
                    writer.flush()
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)