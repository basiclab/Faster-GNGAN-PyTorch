import os

import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

import models.wgangp as models
import common.losses as losses
from common.utils import generate_imgs, infiniteloop, set_seed
from common.score.score import get_inception_and_fid_score


net_G_models = {
    'res32': models.ResGenerator32,
    'res48': models.ResGenerator48,
    'cnn32': models.Generator32,
    'cnn48': models.Generator48,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'res48': models.ResDiscriminator48,
    'cnn32': models.Discriminator32,
    'cnn48': models.Discriminator48,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10'], "dataset")
flags.DEFINE_enum('arch', 'res32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('alpha', 10, "gradient penalty")
flags.DEFINE_enum('loss', 'was', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/WGANGP_CIFAR10_RES', 'logging folder')
flags.DEFINE_bool('record', True, "record inception score and FID score")
flags.DEFINE_string('fid_cache', './data/cifar10_stats.npz', 'FID cache')
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


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())
    norm = torch.norm(torch.flatten(real - fake, start_dim=1), dim=1)
    unit_vec = (real - fake) / norm[:, None, None, None]

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    grad_cosine = torch.sum(grad * unit_vec, dim=[1, 2, 3]) / grad_norm
    loss_norm = torch.mean((grad_norm - 1) ** 2)
    return loss_norm, grad_cosine, grad_norm


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

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    loss_fn = loss_fns[FLAGS.loss]()

    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)
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
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)
    with trange(1, FLAGS.total_steps + 1, dynamic_ncols=True) as pbar:
        for step in pbar:
            # Discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real = next(looper).to(device)
                net_D_real = net_D(real)
                net_D_fake = net_D(fake)
                loss = loss_fn(net_D_real, net_D_fake)
                loss_norm, grad_cosine, grad_norm = cacl_gradient_penalty(
                    net_D, real, fake)
                loss_all = loss + FLAGS.alpha * loss_norm

                optim_D.zero_grad()
                loss_all.backward()
                optim_D.step()

                with torch.no_grad():
                    delta_f = torch.norm(net_D_real - net_D_fake, dim=1)
                    delta_x = torch.norm(
                        torch.flatten(real - fake, start_dim=1), dim=1)
                    slop = delta_f / delta_x
                    slop = slop.max()
                if FLAGS.loss == 'was':
                    loss = -loss
                pbar.set_postfix(loss='%.4f' % loss, slop='%.4f' % slop)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_gp', loss_norm, step)
            writer.add_scalar('grad_norm', grad_norm.max(), step)
            writer.add_scalar('grad_cosine', grad_cosine.min(), step)
            writer.add_scalar('slop', slop, step)

            # Generator
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            loss = loss_fn(net_D(net_G(z)))

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            sched_G.step()
            sched_D.step()
            pbar.update(1)

            if step == 1 or step % FLAGS.sample_step == 0:
                fake = net_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                writer.add_image('sample', grid, step)
                save_image(grid, os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step))

            if step == 1 or step % FLAGS.eval_step == 0:
                torch.save({
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                }, os.path.join(FLAGS.logdir, 'model.pt'))
                if FLAGS.record:
                    imgs = generate_imgs(
                        net_G, device, FLAGS.z_dim, 50000, FLAGS.batch_size)
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
    writer.close()


def main(argv):
    set_seed(FLAGS.seed)
    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
