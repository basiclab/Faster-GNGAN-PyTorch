import os
import json
import math
import warnings

import torch
from absl import flags, app
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange

from source.models import gn_biggan, sn_biggan, gn_gan
from source.losses import HingeLoss
from source.datasets import get_dataset
from source.utils import module_no_grad, infiniteloop, set_seed
from metrics.score.both import get_inception_and_fid_score


archs = {
    'gn-biggan32': (gn_biggan.Generator32, gn_biggan.Discriminator32),
    'sn-biggan32': (sn_biggan.Generator32, sn_biggan.Discriminator32),
    'gn-biggan128': (gn_biggan.Generator128, gn_biggan.Discriminator128),
    'sn-biggan128': (sn_biggan.Generator128, sn_biggan.Discriminator128),
}


datasets = ['cifar10', 'imagenet128', 'imagenet128.hdf5']


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
flags.DEFINE_enum('resume_type', 'last', ['last', 'best'], 'resume type')
# model and training
flags.DEFINE_enum('dataset', 'cifar10', datasets, "select dataset")
flags.DEFINE_enum('arch', 'gn-biggan32', archs.keys(), "model architecture")
flags.DEFINE_integer('ch', 64, 'base channel size of BigGAN')
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('total_steps', 125000, "the number of training steps")
flags.DEFINE_integer('lr_decay_start', 125000, 'linear decay start step')
flags.DEFINE_integer('G_batch_size', 50, "batch size")
flags.DEFINE_integer('D_batch_size', 50, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_float('G_lr', 1e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_float('eps', 1e-8, "for Adam")
flags.DEFINE_multi_float('betas', [0.0, 0.999], "for Adam")
flags.DEFINE_integer('n_dis', 4, "update generator every `n_dis` steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('scale', 1., "scale the output value of discriminator")
flags.DEFINE_float('cr', 0, "weight for consistency regularization")
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_string('logdir', './logs/GN-cGAN_CIFAR10_BIGGAN_0', 'log folder')
flags.DEFINE_string('fid_ref', './stats/cifar10_test.npz', 'FID reference')
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 1000, "evaluation frequency")
flags.DEFINE_integer('save_step', 20000, "saving frequency")
flags.DEFINE_integer('num_images', 10000, "evaluation images")
flags.DEFINE_integer('sample_step', 500, "sampling frequency")
flags.DEFINE_integer('sample_size', 64, "the number of sampling images")
# generate sample
flags.DEFINE_bool('generate', False, 'generate images from pretrain model')


class GeneratorDiscriminator(torch.nn.Module):
    def __init__(self, net_G, net_D):
        super().__init__()
        self.net_G = net_G
        self.net_D = net_D

    def forward(self, z, y_fake, x_real=None, y_real=None):
        if x_real is not None and y_real is not None:
            x_fake = self.net_G(z, y_fake).detach()
            x = torch.cat([x_real, x_fake], dim=0)
            y = torch.cat([y_real, y_fake], dim=0)
            pred = self.net_D(x, y=y)
            net_D_real, net_D_fake = torch.split(
                pred, [x_real.shape[0], x_fake.shape[0]])
            return net_D_real, net_D_fake
        else:
            x_fake = self.net_G(z, y_fake)
            net_D_fake = self.net_D(x_fake, y=y_fake)
            return net_D_fake


class Trainer:
    def __init__(self):
        torch.cuda.set_device(0)
        dataset = get_dataset(FLAGS.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.D_batch_size * FLAGS.D_accumulation * FLAGS.n_dis,
            shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        # model
        Generator, Discriminator = archs[FLAGS.arch]
        self.net_G = Generator(FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim).cuda()
        self.ema_G = Generator(FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim).cuda()
        self.net_D = Discriminator(FLAGS.ch, FLAGS.n_classes).cuda()
        if FLAGS.arch.startswith('gn'):
            self.net_D = gn_gan.GradNorm(self.net_D)
        self.net_GD = GeneratorDiscriminator(self.net_G, self.net_D)

        if FLAGS.parallel:
            self.net_GD = torch.nn.DataParallel(self.net_GD)

        # ema initialize
        self.ema(step=0)

        # loss
        self.loss_fn = HingeLoss(FLAGS.scale)

        # optimizer
        self.optim_G = Adam(
            self.net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
        self.optim_D = Adam(
            self.net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

        # scheduler
        self.sched_G = LambdaLR(self.optim_G, lr_lambda=self.decay_lr)
        self.sched_D = LambdaLR(self.optim_D, lr_lambda=self.decay_lr)

        self.writer = SummaryWriter(FLAGS.logdir)
        if FLAGS.resume:
            if FLAGS.resume_type == 'last':
                ckpt = torch.load(os.path.join(FLAGS.logdir, 'model.pt'))
            else:
                ckpt = torch.load(os.path.join(FLAGS.logdir, 'best_model.pt'))
            self.net_G.load_state_dict(ckpt['net_G'])
            self.net_D.load_state_dict(ckpt['net_D'])
            self.ema_G.load_state_dict(ckpt['ema_G'])
            self.optim_G.load_state_dict(ckpt['optim_G'])
            self.optim_D.load_state_dict(ckpt['optim_D'])
            self.sched_G.load_state_dict(ckpt['sched_G'])
            self.sched_D.load_state_dict(ckpt['sched_D'])
            self.fixed_z = ckpt['fixed_z']
            self.fixed_y = ckpt['fixed_y']
            self.best_IS, self.best_FID = ckpt['best_IS'], ckpt['best_FID']
            self.start = ckpt['step'] + 1
            del ckpt
        else:
            # sample fixed z and y
            os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
            self.fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).cuda()
            self.fixed_z = torch.split(self.fixed_z, FLAGS.G_batch_size, dim=0)
            self.fixed_y = torch.randint(
                FLAGS.n_classes, (FLAGS.sample_size,)).cuda()
            self.fixed_y = torch.split(self.fixed_y, FLAGS.G_batch_size, dim=0)
            with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
                f.write(FLAGS.flags_into_string())
            # sample real data
            real = next(iter(self.dataloader))[0][:FLAGS.sample_size]
            self.writer.add_image('real_sample', make_grid((real + 1) / 2))
            self.writer.flush()
            # initialize
            self.start, self.best_IS, self.best_FID = 1, 0, 999

        if FLAGS.cr > 0:
            self.cr_transforms = transforms.Compose([
                transforms.Lambda(lambda x: (x + 1) / 2),
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0.2, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        D_size = sum(p.data.nelement() for p in self.net_D.parameters())
        G_size = sum(p.data.nelement() for p in self.net_G.parameters())
        print('D params: %d, G params: %d' % (D_size, G_size))

    # ========================= main tasks =========================
    def generate(self):
        _, _, images = self.calc_metrics(self.ema_G)
        path = os.path.join(FLAGS.logdir, 'generate')
        os.makedirs(path, exist_ok=True)
        for i in trange(len(images), dynamic_ncols=True, desc="save images"):
            image = torch.tensor(images[i])
            save_image(image, os.path.join(path, '%d.png' % i))

    def train(self):
        with trange(self.start, FLAGS.total_steps + 1, dynamic_ncols=True,
                    initial=self.start, total=FLAGS.total_steps) as pbar:
            looper = infiniteloop(self.dataloader)
            for step in pbar:
                x, y = next(looper)
                metrics = self.train_step(x, y)

                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, step)
                pbar.set_postfix_str(
                    "loss_real={loss_real:.3f}, "
                    "loss_fake={loss_fake:.3f}".format(**metrics))

                self.ema(step)

                if step == 1 or step % FLAGS.sample_step == 0:
                    self.sample(step)
                if step == 1 or step % FLAGS.eval_step == 0:
                    self.eval(step, pbar)
        self.writer.close()

    # ========================= training functions =========================
    def ema(self, step):
        decay = 0 if step < FLAGS.ema_start else FLAGS.ema_decay
        source_dict = self.net_G.state_dict()
        target_dict = self.ema_G.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def train_step(self, x, y):
        loss_sum = 0
        loss_real_sum = 0
        loss_fake_sum = 0
        loss_cr_sum = 0
        self.net_GD.train()

        # train Discriminator
        x = iter(torch.split(x, FLAGS.D_batch_size))
        y = iter(torch.split(y, FLAGS.D_batch_size))
        for n in range(FLAGS.n_dis):
            self.optim_D.zero_grad()
            for _ in range(FLAGS.D_accumulation):
                x_real, y_real = next(x).cuda(), next(y).cuda()
                z_ = torch.randn(FLAGS.D_batch_size, FLAGS.z_dim).cuda()
                y_ = torch.randint(
                    FLAGS.n_classes, (FLAGS.D_batch_size,)).cuda()
                pred_real, pred_fake = self.net_GD(z_, y_, x_real, y_real)
                loss, loss_real, loss_fake = self.loss_fn(pred_fake, pred_real)
                if FLAGS.cr > 0:
                    loss_cr = self.consistency_loss(x_real, y_real, pred_real)
                else:
                    loss_cr = torch.tensor(0.)
                loss_all = loss + FLAGS.cr * loss_cr
                loss_all = loss_all / FLAGS.D_accumulation
                loss_all.backward()

                loss_sum += loss.cpu().item()
                loss_real_sum += loss_real.cpu().item()
                loss_fake_sum += loss_fake.cpu().item()
                loss_cr_sum += loss_cr.cpu().item()
            self.optim_D.step()

        metrics = {
            'loss': loss_sum / FLAGS.n_dis / FLAGS.D_accumulation,
            'loss_real': loss_real_sum / FLAGS.n_dis / FLAGS.D_accumulation,
            'loss_fake': loss_fake_sum / FLAGS.n_dis / FLAGS.D_accumulation,
            'loss_cr': loss_cr_sum / FLAGS.n_dis / FLAGS.D_accumulation,
        }

        # train Generator
        self.optim_G.zero_grad()
        with module_no_grad(self.net_D):
            for _ in range(FLAGS.G_accumulation):
                z = torch.randn(FLAGS.G_batch_size, FLAGS.z_dim).cuda()
                y = torch.randint(
                    FLAGS.n_classes, (FLAGS.G_batch_size,)).cuda()
                loss = self.loss_fn(self.net_GD(z, y)) / FLAGS.G_accumulation
                loss.backward()
        self.optim_G.step()
        self.sched_G.step()
        self.sched_D.step()

        return metrics

    def eval(self, step, pbar):
        net_IS, net_FID, _ = self.calc_metrics(self.net_G)
        ema_IS, ema_FID, _ = self.calc_metrics(self.ema_G)
        if not math.isnan(ema_FID) and ema_FID < self.best_FID:
            self.best_IS = ema_IS
            self.best_FID = ema_FID
            save_as_best = True
        else:
            save_as_best = False
        ckpt = {
            'net_G': self.net_G.state_dict(),
            'net_D': self.net_D.state_dict(),
            'ema_G': self.ema_G.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'sched_G': self.sched_G.state_dict(),
            'sched_D': self.sched_D.state_dict(),
            'fixed_z': self.fixed_z,
            'fixed_y': self.fixed_y,
            'best_IS': self.best_IS,
            'best_FID': self.best_FID,
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
            'IS': net_IS[0],
            'IS_std': net_IS[1],
            'FID': net_FID,
            'IS_EMA': ema_IS[0],
            'IS_std_EMA': ema_IS[1],
            'FID_EMA': ema_FID
        }
        pbar.write(
            "%d/%d " % (step, FLAGS.total_steps) +
            "IS: %.2f(%.2f), FID: %.2f, IS_EMA: %.2f(%.2f), FID_EMA: %.2f" % (
                metrics['IS'], metrics['IS_std'], metrics['FID'],
                metrics['IS_EMA'], metrics['IS_std_EMA'], metrics['FID_EMA']))
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        self.writer.flush()
        with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
            metrics['step'] = step
            f.write(json.dumps(metrics) + "\n")

        return metrics

    def sample(self, step):
        images_list = []
        with torch.no_grad():
            for x, y in zip(self.fixed_z, self.fixed_y):
                images = self.ema_G(x, y).cpu()
                images_list.append((images + 1) / 2)
            grid = make_grid(torch.cat(images_list, dim=0))
        self.writer.add_image('sample', grid, step)
        save_image(grid, os.path.join(FLAGS.logdir, 'sample', '%d.png' % step))

    # ========================= tool functions =========================
    def calc_metrics(self, net_G):
        if FLAGS.parallel:
            net_G = torch.nn.DataParallel(net_G)
        images = None
        with torch.no_grad():
            for start in trange(0, FLAGS.num_images, FLAGS.G_batch_size,
                                dynamic_ncols=True, leave=False):
                batch_size = min(FLAGS.G_batch_size, FLAGS.num_images - start)
                z = torch.randn(batch_size, FLAGS.z_dim).cuda()
                y = torch.randint(FLAGS.n_classes, size=(batch_size,)).cuda()
                batch_images = net_G(z, y).cpu()
                if images is None:
                    _, C, H, W = batch_images.shape
                    images = torch.zeros((FLAGS.num_images, C, H, W))
                images[start: start + len(batch_images)] = batch_images
        images = (images.numpy() + 1) / 2
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, FLAGS.fid_ref,
            num_images=FLAGS.num_images,
            use_torch=FLAGS.eval_use_torch,
            verbose=True,
            parallel=FLAGS.parallel)
        return (IS, IS_std), FID, images

    def consistency_loss(self, x, y, pred):
        aug_x = x.detach().clone().cpu()
        for idx, img in enumerate(aug_x):
            aug_x[idx] = self.cr_transforms(img)
        aug_x = aug_x.cuda()
        loss = ((self.net_D(aug_x, y=y) - pred) ** 2).mean()
        return loss

    def decay_lr(self, step):
        return 1 - max(step - FLAGS.lr_decay_start, 0) / FLAGS.total_steps


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    set_seed(FLAGS.seed)
    trainer = Trainer()
    if FLAGS.generate:
        trainer.generate()
    else:
        trainer.train()


if __name__ == '__main__':
    app.run(main)
