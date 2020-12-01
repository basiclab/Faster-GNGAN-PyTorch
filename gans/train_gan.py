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

from models import gn_gan, sn_gan
from common.losses import HingeLoss, BCEWithLogits
from common.datasets import get_dataset
from common.score.score import get_inception_and_fid_score
from common.utils import generate_images, save_images, infiniteloop, set_seed


net_G_models = {
    'gn-cnn32': gn_gan.Generator32,
    'gn-cnn48': gn_gan.Generator48,
    'gn-res32': gn_gan.ResGenerator32,
    'gn-res48': gn_gan.ResGenerator48,
    'sn-cnn32': sn_gan.Generator32,
    'sn-cnn48': sn_gan.Generator48,
    'sn-res32': sn_gan.ResGenerator32,
    'sn-res48': sn_gan.ResGenerator48,
}

net_D_models = {
    'gn-cnn32': gn_gan.Discriminator32,
    'gn-cnn48': gn_gan.Discriminator48,
    'gn-res32': gn_gan.ResDiscriminator32,
    'gn-res48': gn_gan.ResDiscriminator48,
    'sn-cnn32': sn_gan.Discriminator32,
    'sn-cnn48': sn_gan.Discriminator48,
    'sn-res32': sn_gan.ResDiscriminator32,
    'sn-res48': sn_gan.ResDiscriminator48,
}

loss_fns = {
    'hinge': HingeLoss,
    'bce': BCEWithLogits,
}


datasets = ['cifar10', 'stl10']


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'cifar10', datasets, "select dataset")
flags.DEFINE_enum('arch', 'gn-res32', net_G_models.keys(), "architecture")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_integer('total_steps', 200000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 0, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 64, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 4e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('scale', 1., "boundary value of hinge loss")
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
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')


def generate():
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    if os.path.isfile(os.path.join(FLAGS.logdir, 'best_model.pt')):
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'best_model.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'model.pt'))

    net_G.load_state_dict(ckpt['net_G'])

    net_G.eval()
    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.batch_size,
        verbose=True)
    save_images(images, os.path.join(FLAGS.logdir, 'generate'))
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, use_torch=FLAGS.eval_use_torch, verbose=True)
    net_G.train()
    print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))


def evaluate(net_G):
    net_G.eval()                # ????
    images = generate_images(
        net_G=net_G,
        z_dim=FLAGS.z_dim,
        num_images=FLAGS.num_images,
        batch_size=FLAGS.batch_size,
        verbose=False)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, use_torch=FLAGS.eval_use_torch, verbose=True)
    del images
    net_G.train()               # ????
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

    aug_real = real.detach().clone().cpu()
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
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    if FLAGS.arch.startswith('gn'):
        net_D = gn_gan.GradNorm(net_D)

    # loss
    loss_fn = loss_fns[FLAGS.loss](FLAGS.scale)

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
        best_IS, best_FID = ckpt['best_IS'], ckpt['best_FID']
        writer = SummaryWriter(FLAGS.logdir)
        del ckpt
    else:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        writer = SummaryWriter(FLAGS.logdir)
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
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
        best_IS, best_FID = 0, 999

    looper = infiniteloop(dataloader)
    with trange(start, FLAGS.total_steps + 1, dynamic_ncols=True,
                initial=start - 1, total=FLAGS.total_steps) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0
            loss_cr_sum = 0

            # Discriminator
            for _ in range(FLAGS.n_dis):
                with torch.no_grad():
                    z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                    fake = net_G(z).detach()
                real, _ = next(looper)
                real = real.to(device)
                pred_real = net_D(real)
                pred_fake = net_D(fake)
                loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                if FLAGS.cr > 0:
                    loss_cr = consistency_loss(net_D, real, pred_real)
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
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            x = net_G(z)
            x.retain_grad()
            loss = loss_fn(net_D(x))

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            avg_grad_norm = torch.norm(torch.flatten(
                (x.grad * x.shape[0]), start_dim=1), p=2, dim=1).mean()
            writer.add_scalar('avg_grad_norm', avg_grad_norm, step)

            # scheduler
            sched_G.step()
            sched_D.step()

            # sample from fixed z
            if step == 1 or step % FLAGS.sample_step == 0:
                with torch.no_grad():
                    fake = net_G(fixed_z).cpu()
                    grid = (make_grid(fake) + 1) / 2
                    writer.add_image('sample', grid, step)
                    save_image(grid, os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step))

            # evaluate IS, FID and save model
            if step == 1 or step % FLAGS.eval_step == 0:
                (IS, IS_std), FID = evaluate(net_G)
                if not math.isnan(FID):
                    save_as_best = (FID < best_FID)
                elif not math.isnan(IS):
                    save_as_best = (IS > best_IS)
                else:
                    save_as_best = False
                if save_as_best:
                    best_IS = best_IS if math.isnan(IS) else IS
                    best_FID = best_FID if math.isnan(FID) else FID
                ckpt = {
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'step': step,
                    'fixed_z': fixed_z,
                    'best_IS': best_IS,
                    'best_FID': best_FID,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                if save_as_best:
                    torch.save(
                        ckpt, os.path.join(FLAGS.logdir, 'best_model.pt'))
                pbar.write(
                    "%6d/%6d "
                    "IS:%6.3f(%.3f), FID:%7.3f" % (
                        step, FLAGS.total_steps,
                        IS, IS_std, FID))
                writer.add_scalar('IS', IS, step)
                writer.add_scalar('IS_std', IS_std, step)
                writer.add_scalar('FID', FID, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    f.write(json.dumps(
                        {
                            'step': step,
                            'IS': IS,
                            'IS_std': IS_std,
                            'FID': FID,
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
