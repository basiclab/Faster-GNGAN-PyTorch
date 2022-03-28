import os
import json

import torch
import torch.optim as optim
from absl import app
from pytorch_gan_metrics import get_inception_score_and_fid
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import trange

from models.gradnorm import (
    normalize_gradient_D, normalize_gradient_G, Rescalable)
from utils import ema, generate_images, infiniteloop, module_no_grad, set_seed
from utils.args import FLAGS, net_G_models, net_D_models, loss_fns
from utils.datasets import get_dataset
from utils.optim import Adam
from utils.regularization import consistency_loss


def evaluate(net_G):
    images = []
    for batch_images in generate_images(net_G,
                                        FLAGS.batch_size_G,
                                        FLAGS.num_images,
                                        FLAGS.z_dim,
                                        FLAGS.n_classes,
                                        verbose=True):
        images.append(batch_images.cpu())
    images = torch.cat(images, dim=0)
    (IS, IS_std), FID = get_inception_score_and_fid(
        images, fid_stats_path=FLAGS.fid_stats, verbose=True)
    del images
    return IS, IS_std, FID


def train(net_D, net_G, optim_D, optim_G, loss_fn, looper, hooks, device):
    loss_sum = 0
    loss_real_sum = 0
    loss_fake_sum = 0
    loss_cr_sum = 0

    # train discriminator
    for _ in range(FLAGS.n_dis):
        if FLAGS.rescale:
            net_D.rescale_model(FLAGS.alpha)

        optim_D.zero_grad()
        for _ in range(FLAGS.accumulation):
            x, y = next(looper)
            x, y = x.to(device), y.to(device)

            z_ = torch.randn(FLAGS.batch_size_D, FLAGS.z_dim).to(device)
            y_ = torch.randint(
                FLAGS.n_classes, (FLAGS.batch_size_D,)).to(device)
            with torch.no_grad():
                x_ = net_G(z_, y_).detach()
            x_all = torch.cat([x, x_], dim=0)
            y_all = torch.cat([y, y_], dim=0)
            f = normalize_gradient_D(net_D, x_all, y=y_all)
            f_real, f_fake = torch.split(f, FLAGS.batch_size_D)

            loss, loss_fake, loss_real = loss_fn(f_fake, f_real)
            if FLAGS.cr > 0:
                loss_cr = consistency_loss(net_D, x, y, f_real)
            else:
                loss_cr = torch.tensor(0.)
            loss_all = (loss + FLAGS.cr * loss_cr) / FLAGS.accumulation
            loss_all.backward()

            loss_sum += loss.detach().item()
            loss_real_sum += loss_real.detach().item()
            loss_fake_sum += loss_fake.detach().item()
            loss_cr_sum += loss_cr.detach().item()
        optim_D.step()

    records = {
        'loss': loss_sum / FLAGS.n_dis,
        'loss/real': loss_real_sum / FLAGS.n_dis / FLAGS.accumulation,
        'loss/fake': loss_fake_sum / FLAGS.n_dis / FLAGS.accumulation,
        'loss/cr': loss_cr_sum / FLAGS.n_dis / FLAGS.accumulation,
    }
    for hook in hooks:
        records[f'feature_norm/{hook.name}'] = hook.norm

    # Generator
    optim_G.zero_grad()
    with module_no_grad(net_D):
        for _ in range(FLAGS.accumulation):
            z = torch.randn(FLAGS.batch_size_G, FLAGS.z_dim).to(device)
            y = torch.randint(
                FLAGS.n_classes, (FLAGS.batch_size_G,)).to(device)
            fake = net_G(z, y)
            f, handle = normalize_gradient_G(net_D, loss_fn, fake, y=y)
            loss = f.mean() / FLAGS.accumulation
            loss.backward()
            handle.remove()
    optim_G.step()

    return records


def main(argv):
    set_seed(FLAGS.seed)
    device = torch.device('cuda:0')

    # wait main process to create hdf5 for small dataset
    dataset = get_dataset(FLAGS.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size_D,
        num_workers=FLAGS.num_workers,
        shuffle=True,
        drop_last=True)

    # model
    net_G = net_G_models[FLAGS.model](FLAGS.z_dim).to(device)
    ema_G = net_G_models[FLAGS.model](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.model]().to(device)

    # loss
    loss_fn = loss_fns[FLAGS.loss]()

    # optimizer
    optim_G = Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=FLAGS.betas)
    optim_D = Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=FLAGS.betas)

    # scheduler
    def decay_rate(step):
        period = max(FLAGS.total_steps - FLAGS.lr_decay_start, 1)
        return 1 - max(step - FLAGS.lr_decay_start, 0) / period
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=decay_rate)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=decay_rate)

    # tensorboard writer
    writer = SummaryWriter(FLAGS.logdir)

    if FLAGS.resume:
        ckpt = torch.load(
            os.path.join(FLAGS.logdir, 'model.pt'), map_location='cpu')
        net_G.load_state_dict(ckpt['net_G'])
        net_D.load_state_dict(ckpt['net_D'])
        ema_G.load_state_dict(ckpt['ema_G'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        sched_G.load_state_dict(ckpt['sched_G'])
        sched_D.load_state_dict(ckpt['sched_D'])
        fixed_z = ckpt['fixed_z'].to(device)
        fixed_y = ckpt['fixed_y'].to(device)
        start = ckpt['step']
        best = ckpt['best']
        del ckpt
    else:
        # sample fixed noises and classes
        fixed_z = torch.randn(
            FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_y = torch.randint(
            FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
        # initialize iteration and best metrics
        start = 0
        best = {
            'IS/best': 0,
            'IS/EMA/best': 0,
            'FID/best': 1000,
            'FID/EMA/best': 1000,
        }
        # initialize ema Generator
        ema(net_G, ema_G, decay=0)

        os.makedirs(os.path.join(FLAGS.logdir, 'sample/net'))
        os.makedirs(os.path.join(FLAGS.logdir, 'sample/ema'))
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
            f.write(FLAGS.flags_into_string())
        samples = [dataset[i][0] for i in range(FLAGS.sample_size)]
        writer.add_image('real_sample', make_grid(samples))
        writer.flush()

    # use forward hook to record feature norm of each layer
    class ScaleHook:
        def __init__(self, name):
            self.name = name

        def __call__(self, module, inputs, outputs):
            with torch.no_grad():
                self.norm = torch.norm(
                    torch.flatten(outputs.detach(), start_dim=1), dim=1).mean()
    hooks = []
    for name, module in net_D.named_modules():
        if isinstance(module, Rescalable):
            hook = ScaleHook(name)
            module.register_forward_hook(hook)
            hooks.append(hook)

    # dataloader loop
    looper = infiniteloop(dataloader)
    progress = trange(start + 1, FLAGS.total_steps + 1,
                      initial=start, total=FLAGS.total_steps,
                      ncols=0, desc='Training')
    for step in progress:
        # a generator update
        records = train(
            net_D, net_G, optim_D, optim_G, loss_fn, looper, hooks, device)

        for name, value in records.items():
            writer.add_scalar(name, value, step)
        progress.set_postfix(
            loss_real='%.3f' % records['loss/real'],
            loss_fake='%.3f' % records['loss/fake'])

        # ema
        if step < FLAGS.ema_start:
            ema(net_G, ema_G, decay=0)
        else:
            ema(net_G, ema_G, decay=FLAGS.ema_decay)

        # learning rate scheduler
        sched_G.step()
        sched_D.step()

        # sample from fixed noise and classes
        if step == 1 or step % FLAGS.sample_step == 0:
            with torch.no_grad():
                imgs_ema = ema_G(fixed_z, fixed_y)
                imgs_net = net_G(fixed_z, fixed_y)
            writer.add_image('sample_ema', make_grid(imgs_ema), step)
            writer.add_image('sample', make_grid(imgs_net), step)
            save_image(
                make_grid(imgs_net),
                os.path.join(FLAGS.logdir, 'sample/net', '%d.png' % step))
            save_image(
                make_grid(imgs_ema),
                os.path.join(FLAGS.logdir, 'sample/net', '%d.png' % step))
            del imgs_ema, imgs_net

        # evaluate IS, FID and save latest model
        if step == 1 or step % FLAGS.eval_step == 0:
            IS, IS_std, FID = evaluate(net_G)
            IS_ema, IS_std_ema, FID_ema = evaluate(ema_G)
            if FID < best['FID/best']:
                best['FID/best'] = FID
                best['IS/best'] = IS
                save_best_model = True
            if FID_ema < best['FID/EMA/best']:
                best['FID/EMA/best'] = FID_ema
                best['IS/EMA/best'] = IS_ema
                save_best_ema_model = True
            ckpt = {
                'net_G': net_G.state_dict(),
                'net_D': net_D.state_dict(),
                'ema_G': ema_G.state_dict(),
                'optim_G': optim_G.state_dict(),
                'optim_D': optim_D.state_dict(),
                'sched_G': sched_G.state_dict(),
                'sched_D': sched_D.state_dict(),
                'fixed_z': fixed_z,
                'fixed_y': fixed_y,
                'best': best,
                'step': step,
            }
            if save_best_model:                             # best non-EMA
                path = os.path.join(FLAGS.logdir, 'best_model.pt')
                torch.save(ckpt, path)
            if save_best_ema_model:                         # best EMA
                path = os.path.join(FLAGS.logdir, 'best_ema_model.pt')
                torch.save(ckpt, path)
            if step == 1 or step % FLAGS.save_step == 0:    # period save
                path = os.path.join(FLAGS.logdir, '%06d.pt' % step)
                torch.save(ckpt, path)
            path = os.path.join(FLAGS.logdir, 'model.pt')   # latest save
            torch.save(ckpt, path)
            metrics = {
                'step': step,
                'IS': IS, 'IS/std': IS_std,
                'IS/EMA': IS_ema, 'IS/EMA/std': IS_std_ema,
                'FID': FID,
                'FID/EMA': FID_ema,
                **best,
            }
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            writer.flush()
            with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                f.write(json.dumps(metrics) + "\n")
            progress.write(", ".join([
                f"{step:6d}/{FLAGS.total_steps:6d}",
                f"IS: {IS:6.3f}({IS_std:.3f})",
                f"IS/EMA: {IS_ema:6.3f}({IS_std_ema:.3f})",
                f"FID: {FID:.3f}",
                f"FID/EMA: {FID_ema:.3f}",
            ]))
    progress.close()
    writer.close()


if __name__ == '__main__':
    app.run(main)
