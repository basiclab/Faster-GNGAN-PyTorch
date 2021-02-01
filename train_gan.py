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

from source.models import sn_gan, gn_gan
from source.losses import HingeLoss, BCEWithLogits, BCEWithoutLogits
from source.datasets import get_dataset
from source.utils import module_no_grad, infiniteloop, set_seed
from metrics.score.both import get_inception_and_fid_score


archs = {
    'gn-cnn32': (gn_gan.Generator32, gn_gan.Discriminator32),
    'sn-cnn32': (sn_gan.Generator32, sn_gan.Discriminator32),
    'gn-cnn48': (gn_gan.Generator48, gn_gan.Discriminator48),
    'sn-cnn48': (sn_gan.Generator48, sn_gan.Discriminator48),
    'gn-res32': (gn_gan.ResGenerator32, gn_gan.ResDiscriminator32),
    'sn-res32': (sn_gan.ResGenerator32, sn_gan.ResDiscriminator32),
    'gn-res48': (gn_gan.ResGenerator48, gn_gan.ResDiscriminator48),
    'sn-res48': (sn_gan.ResGenerator48, sn_gan.ResDiscriminator48),
    'gn-res128': (gn_gan.ResGenerator128, gn_gan.ResDiscriminator128),
    'sn-res128': (sn_gan.ResGenerator128, sn_gan.ResDiscriminator128),
}

loss_fns = {
    'hinge': HingeLoss,
    'bce': BCEWithLogits,
    'bce-without-logits': BCEWithoutLogits,
}

datasets = [
    'cifar10', 'stl10',
    'imagenet128', 'imagenet128.hdf5',
    'celebhq128', 'celebhq128.hdf5',
    'lsun_church_outdoor', 'lsun_church_outdoor.hdf5'
]


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
flags.DEFINE_enum('resume_type', 'last', ['last', 'best'], 'resume type')
# model and training
flags.DEFINE_enum('dataset', 'cifar10', datasets, "select dataset")
flags.DEFINE_enum('arch', 'gn-res32', archs.keys(), "model architecture")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_integer('total_steps', 100000, "the number of training steps")
flags.DEFINE_integer('lr_decay_start', 100000, 'linear decay start step')
flags.DEFINE_integer('batch_size', 16, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('G_accumulation', 1, 'gradient accumulation for G')
flags.DEFINE_integer('D_accumulation', 1, 'gradient accumulation for D')
flags.DEFINE_float('G_lr', 2e-4, "generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update generator every `n_dis` steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('scale', 1., "scale the output value of discriminator")
flags.DEFINE_float('cr', 0, "weight for consistency regularization")
flags.DEFINE_bool('parallel', False, 'multi-gpu training')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 5000, "start step for ema")
# logging
flags.DEFINE_string('logdir', './logs/GN-GAN_CIFAR10_RES_0', 'log folder')
flags.DEFINE_string('fid_ref', './stats/celebhq_val128.npz', 'FID reference')
flags.DEFINE_bool('eval_use_torch', False, 'evaluate on GPU')
flags.DEFINE_bool('eval_in_eval_mode', False, 'evaluate in eval mode')
flags.DEFINE_integer('eval_step', 5000, "evaluation frequency")
flags.DEFINE_integer('save_step', 20000, "saving frequency")
flags.DEFINE_integer('num_images', 10000, "evaluation images")
flags.DEFINE_integer('sample_step', 500, "sampling frequency")
flags.DEFINE_integer('sample_size', 64, "the number of sampling images")
# generate
flags.DEFINE_bool('generate', False, 'generate images from pretrain model')
flags.DEFINE_bool('generate_use_eam', False, 'use ema model')

device = torch.device('cuda:0')


class GeneratorDiscriminator(torch.nn.Module):
    def __init__(self, net_G, net_D):
        super().__init__()
        self.net_G = net_G
        self.net_D = net_D

    def forward(self, z, x_real=None):
        if x_real is not None:
            x_fake = self.net_G(z).detach()
            x = torch.cat([x_real, x_fake], dim=0)
            pred = self.net_D(x)
            net_D_real, net_D_fake = torch.split(
                pred, [x_real.shape[0], x_fake.shape[0]])
            return net_D_real, net_D_fake
        else:
            x_fake = self.net_G(z)
            net_D_fake = self.net_D(x_fake)
            return net_D_fake


class Trainer:
    def __init__(self):
        torch.cuda.set_device(0)
        dataset = get_dataset(FLAGS.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size * FLAGS.D_accumulation * FLAGS.n_dis,
            shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

        # model
        Generator, Discriminator = archs[FLAGS.arch]
        self.net_G = Generator(FLAGS.z_dim).to(device)
        self.ema_G = Generator(FLAGS.z_dim).to(device)
        self.net_D = Discriminator().to(device)
        if FLAGS.arch.startswith('gn'):
            self.net_D = gn_gan.GradNorm(self.net_D)
        self.net_GD = GeneratorDiscriminator(self.net_G, self.net_D)

        if FLAGS.parallel:
            self.net_GD = torch.nn.DataParallel(self.net_GD)

        # ema
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
            self.best_IS, self.best_FID = ckpt['best_IS'], ckpt['best_FID']
            self.start = ckpt['step'] + 1
            del ckpt
        else:
            os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
            self.fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).cuda()
            self.fixed_z = torch.split(self.fixed_z, FLAGS.batch_size, dim=0)
            with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
                f.write(FLAGS.flags_into_string())
            # sample real data
            real = next(iter(self.dataloader))[0][:FLAGS.sample_size]
            self.writer.add_image('real_sample', make_grid((real + 1) / 2))
            self.writer.flush()
            # start value
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
        if FLAGS.generate_use_eam:
            _, _, images = self.calc_metrics(self.ema_G)
        else:
            _, _, images = self.calc_metrics(self.net_G)
        path = os.path.join(FLAGS.logdir, 'generate')
        for i in trange(len(images), dynamic_ncols=True, desc="save images"):
            image = torch.tensor(images[i])
            save_image(image, os.path.join(path, '%d.png' % i))

    def train(self):
        with trange(self.start, FLAGS.total_steps + 1, dynamic_ncols=True,
                    initial=self.start, total=FLAGS.total_steps) as pbar:
            looper = infiniteloop(self.dataloader)
            for step in pbar:
                x, _ = next(looper)
                metrics = self.train_step(x)

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

    def train_step(self, x):
        loss_sum = 0
        loss_real_sum = 0
        loss_fake_sum = 0
        loss_cr_sum = 0
        self.net_GD.train()

        # Discriminator
        x = iter(torch.split(x, FLAGS.batch_size))
        for _ in range(FLAGS.n_dis):
            self.optim_D.zero_grad()
            for _ in range(FLAGS.D_accumulation):
                real = next(x).to(device)
                z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).cuda()
                pred_real, pred_fake = self.net_GD(z, real)
                loss, loss_real, loss_fake = self.loss_fn(pred_fake, pred_real)
                if FLAGS.cr > 0:
                    loss_cr = self.consistency_loss(real, pred_real)
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

        # Generator
        self.optim_G.zero_grad()
        with module_no_grad(self.net_D):
            for _ in range(FLAGS.G_accumulation):
                z = torch.randn(2 * FLAGS.batch_size, FLAGS.z_dim).cuda()
                loss = self.loss_fn(self.net_GD(z)) / FLAGS.G_accumulation
                loss.backward()
        self.optim_G.step()

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
            ", ".join('%s:%.2f' % (k, v) for k, v in metrics.items()))
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
            for z in self.fixed_z:
                images = self.ema_G(z).cpu()
                images_list.append((images + 1) / 2)
            grid = make_grid(torch.cat(images_list, dim=0))
        self.writer.add_image('sample', grid, step)
        save_image(grid, os.path.join(FLAGS.logdir, 'sample', '%d.png' % step))

    # ========================= tool functions =========================
    def calc_metrics(self, net_G):
        if FLAGS.parallel:
            net_G = torch.nn.DataParallel(net_G)
        if FLAGS.eval_in_eval_mode:
            net_G.eval()
        images = None
        with torch.no_grad():
            for start in trange(0, FLAGS.num_images, FLAGS.batch_size,
                                dynamic_ncols=True, leave=False):
                batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
                z = torch.randn(batch_size, FLAGS.z_dim).cuda()
                batch_images = net_G(z).cpu()
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

    def consistency_loss(self, x, pred):
        aug_x = x.detach().clone().cpu()
        for idx, img in enumerate(aug_x):
            aug_x[idx] = self.cr_transforms(img)
        aug_x = aug_x.cuda()
        loss = ((self.net_D(aug_x) - pred) ** 2).mean()
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
