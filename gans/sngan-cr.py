import os
from copy import deepcopy

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

import models.sngan as models
import common.losses as losses
from common.utils import generate_imgs, infiniteloop, set_seed
from common.score.score import get_inception_and_fid_score


net_G_models = {
    'res32': models.ResGenerator32,
    'res48': models.ResGenerator48,
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
    'res128': models.ResGenerator128,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'res48': models.ResDiscriminator48,
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
    'res128': models.ResDiscriminator128,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}


FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum(
    'dataset', 'cifar10', ['cifar10', 'stl10', 'imagenet'], "dataset")
flags.DEFINE_enum('arch', 'res32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_bool('scheduler', True, 'apply linearly LR decay')
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_float('alpha', 5, "Consistency penalty")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/SNGAN_CIFAR10_RES_CR', 'log folder')
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', './stats/cifar10_test.npz', 'FID cache')
# generate
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
            x = net_G(z).cpu()
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
    if FLAGS.dataset == 'stl10':
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((48, 48)),
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

    consistency_transforms = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),
        transforms.ToPILImage(mode='RGB'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def consistency_transform_func(images):
        images = deepcopy(images)
        for idx, img in enumerate(images):
            images[idx] = consistency_transforms(img)
        return images

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    if FLAGS.parallel:
        net_G = torch.nn.DataParallel(net_G)
        net_D = torch.nn.DataParallel(net_D)
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
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    aug_real = consistency_transform_func(real)

    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)
    grid = (make_grid(aug_real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('augment_real_sample', grid)
    writer.flush()

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real, _ = next(looper)
                real = real.to(device)
                augment_real = consistency_transform_func(real)
                real = real.to(device)
                augment_real = augment_real.to(device)

                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss, loss_real, loss_fake = loss_fn(net_D_real, net_D_fake)

                loss_cs = ((net_D_real - net_D(augment_real))**2).mean()
                loss_all = loss + FLAGS.alpha * loss_cs

                optim_D.zero_grad()
                loss_all.backward()
                optim_D.step()

                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss)

            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_cs', loss_cs, step)
            writer.add_scalar('loss_real', loss_real, step)
            writer.add_scalar('loss_fake', loss_fake, step)

            # Generator
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            x = net_G(z)
            loss = loss_fn(net_D(x))

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            if FLAGS.scheduler:
                sched_G.step()
                sched_D.step()

            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                with torch.no_grad():
                    fake = net_G(sample_z).cpu()
                    grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                writer.flush()
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
                if FLAGS.scheduler:
                    ckpt.update({
                        'sched_G': sched_G.state_dict(),
                        'sched_D': sched_D.state_dict(),
                    })
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device,
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