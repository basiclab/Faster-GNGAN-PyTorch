import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from models import biggan, gngan
from common import losses
from common.utils import generate_conditional_imgs, set_seed, infiniteloop
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
flags.DEFINE_integer('total_steps', 125000, "total number of training steps")
flags.DEFINE_integer('batch_size', 50, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for net_G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_integer('n_dis', 4, "update Generator every this steps")
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.999], "for Adam")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_bool('GN', False, "enable gradient penalty")
flags.DEFINE_bool('scheduler', False, 'apply linear learing rate decay')
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_bool('ema', True, 'exponential moving average params')
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/BIGGAN/CIFAR10/0', 'log folder')
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
flags.DEFINE_bool('record', True, "record inception score and FID score")
# generate sample
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay)
        )


def set_grad(m: torch.nn.Module, require_grad: bool):
    for param in m.parameters():
        param.requires_grad_(require_grad)


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


def evaluate(net_G, writer, pbar, step, suffix=""):
    imgs = generate_conditional_imgs(
        net_G, device, FLAGS.n_classes, FLAGS.z_dim, FLAGS.num_images,
        FLAGS.batch_size)
    is_score, fid_score = get_inception_and_fid_score(
        imgs, device, FLAGS.fid_cache, verbose=True)
    pbar.write("%s/%s Inception Score: %.3f(%.5f), FID Score: %6.3f %s" % (
        step, FLAGS.total_steps, is_score[0], is_score[1], fid_score, suffix))
    if len(suffix) > 0:
        suffix = "/" + suffix
    writer.add_scalar('inception_score' + suffix, is_score[0], step)
    writer.add_scalar('inception_score_std' + suffix, is_score[1], step)
    writer.add_scalar('fid_score' + suffix, fid_score, step)
    writer.flush()


def train():
    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'imagenet':
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
    net_G = biggan.Generator32(
        FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim).to(device)
    if FLAGS.GN:
        net_D = biggan.Discriminator32(
            FLAGS.ch, FLAGS.n_classes, sn=lambda x: x).to(device)
        net_D = gngan.GradNorm(net_D)
    else:
        net_D = biggan.Discriminator32(
            FLAGS.ch, FLAGS.n_classes).to(device)
    net_GD = biggan.GenDis(net_G, net_D)

    if FLAGS.parallel:
        net_GD = torch.nn.DataParallel(net_GD)
    loss_fn = loss_fns[FLAGS.loss]()

    # ema
    if FLAGS.ema:
        net_G_ema = biggan.Generator32(
            FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim).to(device)
        ema(net_G, net_G_ema, decay=0)

    # optimizer
    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    # scheduler
    if FLAGS.scheduler:
        sched_G = optim.lr_scheduler.LambdaLR(
            optim_G, lambda step: 1 - step / FLAGS.total_steps)
        sched_D = optim.lr_scheduler.LambdaLR(
            optim_D, lambda step: 1 - step / FLAGS.total_steps)

    # sample fixed z and y
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(FLAGS.logdir)
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

            set_grad(net_G, False)
            set_grad(net_D, True)
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

            set_grad(net_G, True)
            set_grad(net_D, False)
            net_G.train()
            optim_G.zero_grad()
            for _ in range(FLAGS.G_accumulation):
                z_rand.normal_()
                y_rand.random_(FLAGS.n_classes)
                loss = loss_fn(net_GD(z_rand, y_rand))
                loss = loss / float(FLAGS.G_accumulation)
                loss.backward()
            optim_G.step()

            # scheduler
            if FLAGS.scheduler:
                sched_G.step()
                sched_D.step()

            if FLAGS.ema:
                if step < FLAGS.ema_start:
                    decay = 0
                else:
                    decay = FLAGS.ema_decay
                ema(net_G, net_G_ema, decay)

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
                if FLAGS.ema:
                    ckpt.update({
                        'net_G_ema': net_G_ema.state_dict(),
                    })
                if FLAGS.scheduler:
                    ckpt.update({
                        'sched_G': sched_G.state_dict(),
                        'sched_D': sched_D.state_dict(),
                    })
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))

                if FLAGS.record:
                    if FLAGS.ema:
                        evaluate(net_G_ema, writer, pbar, step, suffix="ema")
                    else:
                        evaluate(net_G, writer, pbar, step, suffix="")
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
