import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from models import sn_cgan, gn_cgan, gn_gan
from common.losses import loss_fns
from common.score.score import get_inception_and_fid_score
from common.utils import (
    ema, generate_conditional_imgs, generate_and_save, module_no_grad,
    infiniteloop, set_seed)


FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'imagenet128'], "dataset")
flags.DEFINE_enum('arch', 'res32', sn_cgan.generators.keys(), "architecture")
flags.DEFINE_enum('norm', 'SN', ['GN', 'SN'], "normalization techniques")
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 0, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for net_G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/SN-cGAN_CIFAR10_0', 'log folder')
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
# generate sample
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    if FLAGS.norm == 'GN':
        Generator = gn_cgan.generators[FLAGS.arch]
        net_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
    if FLAGS.norm == 'SN':
        Generator = sn_cgan.generators[FLAGS.arch]
        net_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
    net_G.load_state_dict(torch.load(FLAGS.pretrain)['net_G'])

    generate_and_save(
        net_G,
        FLAGS.output_dir,
        FLAGS.z_dim,
        FLAGS.num_images,
        FLAGS.batch_size)


def evaluate(net_G, writer, pbar, step):
    imgs = generate_conditional_imgs(
        net_G,
        FLAGS.n_classes,
        FLAGS.z_dim,
        FLAGS.num_images,
        FLAGS.batch_size)
    (is_mean, is_std), fid_score = get_inception_and_fid_score(
        imgs, FLAGS.fid_cache, parallel=FLAGS.parallel)
    pbar.write("%s/%s Inception Score: %.3f(%.5f), FID Score: %6.3f" % (
        step, FLAGS.total_steps, is_mean, is_std, fid_score))
    writer.add_scalar('inception_score', is_mean, step)
    writer.add_scalar('inception_score_std', is_std, step)
    writer.add_scalar('fid_score', fid_score, step)
    writer.flush()


def train():
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'imagenet128':
        dataset = datasets.ImageFolder(
            './data/ILSVRC2012/train',
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    # model
    if FLAGS.norm == 'GN':
        Generator = gn_cgan.generators[FLAGS.arch]
        Discriminator = gn_cgan.discriminators[FLAGS.arch]
        net_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
        net_D = Discriminator(FLAGS.n_classes).to(device)
        net_D = gn_gan.GradNorm(net_D)
        ema_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
        net_GD = gn_cgan.GenDis(net_G, net_D)
    if FLAGS.norm == 'SN':
        Generator = sn_cgan.generators[FLAGS.arch]
        Discriminator = sn_cgan.discriminators[FLAGS.arch]
        net_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
        net_D = Discriminator(FLAGS.n_classes).to(device)
        ema_G = Generator(FLAGS.n_classes, FLAGS.z_dim).to(device)
        net_GD = sn_cgan.GenDis(net_G, net_D)

    if FLAGS.parallel:
        net_GD = torch.nn.DataParallel(net_GD)

    # ema
    ema(net_G, ema_G, decay=0)

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

    # sample fixed z and y
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(FLAGS.logdir)
    writer_ema = SummaryWriter(FLAGS.logdir + "_ema")
    fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    fixed_z = torch.split(fixed_z, FLAGS.batch_size, dim=0)
    fixed_y = torch.randint(FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
    fixed_y = torch.split(fixed_y, FLAGS.batch_size, dim=0)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    # sample real data
    real = []
    for x, _ in dataloader:
        real.append(x)
        if len(real) * FLAGS.batch_size >= FLAGS.sample_size:
            real = torch.cat(real, dim=0)[:FLAGS.sample_size]
            break
    grid = (make_grid(real) + 1) / 2
    writer.add_image('real_sample', grid)
    writer.flush()

    z_rand = torch.zeros(
        FLAGS.batch_size, FLAGS.z_dim, dtype=torch.float).to(device)
    y_rand = torch.zeros(
        FLAGS.batch_size, dtype=torch.long).to(device)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            loss_list = []
            loss_real_list = []
            loss_fake_list = []

            # Discriminator
            net_D.train()
            for _ in range(FLAGS.n_dis):
                optim_D.zero_grad()
                for _ in range(FLAGS.D_accumulation):
                    x_real, y_real = next(looper)
                    x_real, y_real = x_real.to(device), y_real.to(device)
                    z_rand.normal_()
                    y_rand.random_(FLAGS.n_classes)
                    pred_real, pred_fake = net_GD(
                        z_rand, y_rand, x_real, y_real)
                    loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                    loss = loss / float(FLAGS.D_accumulation)
                    loss.backward()
                    loss_list.append(loss.detach())
                    loss_real_list.append(loss_real.detach())
                    loss_fake_list.append(loss_fake.detach())
                optim_D.step()

            loss = torch.mean(torch.stack(loss_list))
            loss_real = torch.mean(torch.stack(loss_real_list))
            loss_fake = torch.mean(torch.stack(loss_fake_list))
            if FLAGS.loss == 'was':
                loss = -loss
            pbar.set_postfix(
                loss='%.4f' % loss,
                loss_real='%.4f' % loss_real,
                loss_fake='%.4f' % loss_fake)

            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('loss_real', loss_real.item(), step)
            writer.add_scalar('loss_fake', loss_fake.item(), step)

            # Generator
            net_G.train()
            optim_G.zero_grad()
            for _ in range(FLAGS.G_accumulation):
                z_rand.normal_()
                y_rand.random_(FLAGS.n_classes)
                with module_no_grad(net_D):
                    loss = loss_fn(net_GD(z_rand, y_rand))
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

            if step == 1 or step % FLAGS.sample_step == 0:
                fake_imgs = []
                with torch.no_grad():
                    for fixed_z_batch, fixed_y_batch in zip(fixed_z, fixed_y):
                        fake = net_G(fixed_z_batch, fixed_y_batch).cpu()
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
                    'ema_G': ema_G.state_dict(),
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.parallel:
                    eval_net_G = torch.nn.DataParallel(net_G)
                    eval_ema_G = torch.nn.DataParallel(ema_G)
                else:
                    eval_net_G = net_G
                    eval_ema_G = ema_G
                pbar.write('Evalutation')
                evaluate(eval_net_G, writer, pbar, step)
                pbar.write('Evalutation(ema)')
                evaluate(eval_ema_G, writer_ema, pbar, step)
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
