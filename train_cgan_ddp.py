import os
import json
import math
import datetime

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import Process
from absl import flags, app
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm

from source.models import gn_biggan
from source.losses import HingeLoss
from source.datasets import get_dataset
from source.utils import ema, module_no_grad, set_seed
from metrics.score.both import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid)


net_G_models = {
    'gn-biggan128': gn_biggan.Generator128,
}

net_D_models = {
    'gn-biggan128': gn_biggan.Discriminator128,
}

net_GD_models = {
    'gn-biggan128': gn_biggan.GenDis
}

datasets = ['imagenet.128.hdf5']


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'imagenet.128.hdf5', datasets, "dataset")
flags.DEFINE_enum('arch', 'gn-biggan128', net_G_models.keys(), "architecture")
flags.DEFINE_integer('ch', 96, 'base channel size of BigGAN')
flags.DEFINE_integer('n_classes', 10, 'the number of classes in dataset')
flags.DEFINE_integer('total_steps', 125000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 125000, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 1024, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('accumulation', 1, 'gradient accumulation steps')
flags.DEFINE_float('G_lr', 1e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 4e-4, "Discriminator learning rate")
flags.DEFINE_float('eps', 1e-6, "for Adam")
flags.DEFINE_multi_float('betas', [0.0, 0.999], "for Adam")
flags.DEFINE_integer('n_dis', 4, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 1000, "start step for ema")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 1000, "evaluate FID and Inception Score")
flags.DEFINE_integer('save_step', 10000, "save model every this step")
flags.DEFINE_integer('num_images', 10000, 'the number of generated images')
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string(
    'logdir', './logs/GN-cGAN_IMAGENET128_BIGGAN_0', 'log folder')
flags.DEFINE_string('fid_stats', './stats/imagenet.train.128.npz', 'FID cache')
# generate sample
flags.DEFINE_bool('generate', False, 'generate images from pretrain model')
flags.DEFINE_string('output', None, 'path to output directory')
# distributed
flags.DEFINE_string('port', '55556', 'distributed port')


def image_generator(net_G):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = FLAGS.batch_size // world_size
    local_num_images = FLAGS.num_images // world_size
    with torch.no_grad():
        for _ in range(0, local_num_images, local_batch_size):
            z = torch.randn(local_batch_size, FLAGS.z_dim).to(rank)
            y = torch.randint(FLAGS.n_classes, (local_batch_size,)).to(rank)
            fake = (net_G(z, y) + 1) / 2
            fake_list = [torch.empty_like(fake) for _ in range(world_size)]
            dist.all_gather(fake_list, fake)
            if rank == 0:
                yield torch.cat(fake_list, dim=0).cpu()
    del fake, fake_list


def generate(rank, world_size):
    device = torch.device('cuda:%d' % rank)

    ckpt = torch.load(
        os.path.join(FLAGS.logdir, 'model.pt'), map_location='cpu')
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G.load_state_dict(ckpt['net_G'])
    net_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_G)
    net_G = DistributedDataParallel(
        net_G, device_ids=[rank], output_device=rank)

    # generate fixed sample
    with torch.no_grad():
        fixed_z = torch.cat(ckpt['fixed_z'], dim=0).to(device)
        fixed_z = torch.split(fixed_z, len(fixed_z) // world_size, dim=0)
        fake = (net_G(fixed_z[rank]) + 1) / 2
        fake_list = [torch.empty_like(fake) for _ in range(world_size)]
        dist.all_gather(fake_list, fake)
        if rank == 0:
            save_image(torch.cat(fake_list, dim=0), 'fixed_sample.png')

    # generate images for calculating IS and FID
    generator = image_generator(net_G)
    if rank == 0:
        if FLAGS.output is not None:
            root = FLAGS.output
        else:
            root = os.path.join(FLAGS.logdir, 'output')
        os.makedirs(root, exist_ok=True)
        pbar = tqdm(
            total=FLAGS.num_images, ncols=0, leave=False, desc="save_images")
        counter = 0
        for batch_images in generator:
            for image in batch_images:
                if counter < FLAGS.num_images:
                    save_image(image, os.path.join(root, '%d.png' % counter))
                    counter += 1
                    pbar.update()
        pbar.close()
        (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
            root, FLAGS.fid_stats,
            use_torch=FLAGS.eval_use_torch, verbose=True)
        print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))
    del ckpt, net_G


def evaluate(net_G):
    if dist.get_rank() != 0:
        image_generator(net_G)
        (IS, IS_std), FID = (None, None), None
    else:
        images = []
        for batch_images in image_generator(net_G):
            images.append(batch_images)
        images = torch.cat(images, dim=0)
        (IS, IS_std), FID = get_inception_score_and_fid(
            images,
            fid_stats_path=FLAGS.fid_stats,
            num_images=len(images),
            use_torch=FLAGS.eval_use_torch,
            parallel=FLAGS.parallel,
            verbose=True)
        del images
    dist.barrier()
    return (IS, IS_std), FID


def infiniteloop(dataloader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            yield x, y


def train(rank, world_size):
    device = torch.device('cuda:%d' % rank)

    dataset = get_dataset(FLAGS.dataset)
    sampler = torch.utils.data.DistributedSampler(
        dataset, seed=FLAGS.seed, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size * FLAGS.n_dis * FLAGS.accumulation,
        sampler=sampler,
        num_workers=FLAGS.num_workers,
        drop_last=True)
    looper = infiniteloop(dataloader, sampler)

    # model
    net_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    net_G = net_G.to(device)
    ema_G = net_G_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes, FLAGS.z_dim)
    ema_G = ema_G.to(device)
    net_D = net_D_models[FLAGS.arch](FLAGS.ch, FLAGS.n_classes).to(device)
    net_GD = net_GD_models[FLAGS.arch](net_G, net_D)

    if rank == 0:
        gn_biggan.res128_weights_init(net_G)
        gn_biggan.res128_weights_init(net_D)

    dist.barrier()

    # loss
    loss_fn = HingeLoss()

    # optimizer
    optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    # scheduler
    def decay_rate(step):
        period = max(FLAGS.total_steps - FLAGS.lr_decay_start, 1)
        return 1 - max(step - FLAGS.lr_decay_start, 0) / period
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=decay_rate)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=decay_rate)

    local_batch_size = FLAGS.batch_size // world_size

    if rank == 0:
        writer = SummaryWriter(FLAGS.logdir)
        D_size = 0
        for param in net_D.parameters():
            D_size += param.data.nelement()
        G_size = 0
        for param in net_G.parameters():
            G_size += param.data.nelement()
        print('D params: %d, G params: %d' % (D_size, G_size))

    if FLAGS.resume:
        ckpt = torch.load(
            os.path.join(FLAGS.logdir, 'model.pt'), map_location='cpu')
        net_G.load_state_dict(ckpt['net_G'])
        net_D.load_state_dict(ckpt['net_D'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        sched_G.load_state_dict(ckpt['sched_G'])
        sched_D.load_state_dict(ckpt['sched_D'])
        ema_G.load_state_dict(ckpt['ema_G'])
        fixed_z = ckpt['fixed_z'].to(device)
        fixed_y = ckpt['fixed_y'].to(device)
        # start value
        start = ckpt['step'] + 1
        best_IS, best_FID = ckpt['best_IS'], ckpt['best_FID']
        del ckpt
    else:
        # sample fixed z and y
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_z = torch.split(fixed_z, FLAGS.sample_size // world_size, dim=0)
        fixed_y = torch.randint(
            FLAGS.n_classes, (FLAGS.sample_size,)).to(device)
        fixed_y = torch.split(fixed_y, FLAGS.sample_size // world_size, dim=0)
        # start value
        start, best_IS, best_FID = 1, 0, 999

        if rank == 0:
            os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
            with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
                f.write(FLAGS.flags_into_string())
            real_sample = None
            for x, _ in dataloader:
                if real_sample is None:
                    real_sample = x
                else:
                    real_sample = torch.cat([real_sample, x])
                if real_sample.size(0) >= FLAGS.sample_size:
                    real_sample = real_sample[:FLAGS.sample_size]
                    break
            writer.add_image('real_sample', make_grid((real_sample + 1) / 2))
            writer.flush()

    net_GD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_GD)
    net_GD = DistributedDataParallel(
        net_GD,
        device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # ema
    ema(net_G, ema_G, decay=0)

    disable_progress = (rank != 0)
    with trange(start, FLAGS.total_steps + 1,
                initial=start - 1, total=FLAGS.total_steps,
                disable=disable_progress, ncols=0) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0

            x, y = next(looper)
            x = iter(torch.split(x, local_batch_size))
            y = iter(torch.split(y, local_batch_size))
            # Discriminator
            for n in range(FLAGS.n_dis):
                optim_D.zero_grad()
                for _ in range(FLAGS.accumulation):
                    x_real, y_real = next(x).to(device), next(y).to(device)
                    z_ = torch.randn(
                        local_batch_size, FLAGS.z_dim, device=device)
                    y_ = torch.randint(
                        FLAGS.n_classes, (local_batch_size,), device=device)
                    pred_real, pred_fake = net_GD(z_, y_, x_real, y_real)
                    loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                    loss = loss / FLAGS.accumulation
                    loss.backward()

                    loss_sum += loss.cpu().item()
                    loss_real_sum += loss_real.cpu().item()
                    loss_fake_sum += loss_fake.cpu().item()
                optim_D.step()

            loss = loss_sum / FLAGS.n_dis
            loss_real = loss_real_sum / FLAGS.n_dis / FLAGS.accumulation
            loss_fake = loss_fake_sum / FLAGS.n_dis / FLAGS.accumulation

            if rank == 0:
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('loss_real', loss_real, step)
                writer.add_scalar('loss_fake', loss_fake, step)

                pbar.set_postfix(
                    loss_real='%.3f' % loss_real,
                    loss_fake='%.3f' % loss_fake)

            # Generator
            optim_G.zero_grad()
            with module_no_grad(net_D):
                for _ in range(FLAGS.accumulation):
                    z = torch.randn(
                        local_batch_size, FLAGS.z_dim, device=device)
                    y = torch.randint(
                        FLAGS.n_classes, (local_batch_size,), device=device)
                    loss = loss_fn(net_GD(z, y)) / FLAGS.accumulation
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
                with torch.no_grad():
                    fake = (ema_G(fixed_z[rank], fixed_y[rank]) + 1) / 2
                fake_list = [torch.empty_like(fake) for _ in range(world_size)]
                dist.all_gather(fake_list, fake)
                if rank == 0:
                    fake = torch.cat(fake_list, dim=0).cpu()
                    grid = make_grid(fake)
                    writer.add_image('sample', grid, step)
                    save_image(
                        grid,
                        os.path.join(FLAGS.logdir, 'sample', '%d.png' % step))
                del fake, fake_list

            if step == 1 or step % FLAGS.eval_step == 0:
                (net_IS, net_IS_std), net_FID = evaluate(net_G)
                (ema_IS, ema_IS_std), ema_FID = evaluate(ema_G)

                if rank != 0:
                    continue

                if not math.isnan(ema_FID) and not math.isnan(best_FID):
                    save_as_best = (ema_FID < best_FID)
                else:
                    save_as_best = (ema_IS > best_IS)
                if save_as_best:
                    best_IS = ema_IS
                    best_FID = best_FID
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
                    'best_IS': best_IS,
                    'best_FID': best_FID,
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
                    'IS': net_IS,
                    'IS_std': net_IS_std,
                    'FID': net_FID,
                    'IS_EMA': ema_IS,
                    'IS_std_EMA': ema_IS_std,
                    'FID_EMA': ema_FID,
                }
                pbar.write(
                    "{}/{} ".format(step, FLAGS.total_steps) +
                    "IS: {IS:6.3f}({IS_std:.3f}), FID: {FID:.3f}, "
                    "IS_EMA: {IS_EMA:6.3f}({IS_std_EMA:.3f}), "
                    "FID_EMA: {FID_EMA:.3f}, ".format(**metrics))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def initialize_process(rank, world_size):
    set_seed(FLAGS.seed)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = FLAGS.port

    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=30),
                            world_size=world_size, rank=rank)
    print("Node %d is initialized" % rank)

    if FLAGS.generate:
        generate(rank, world_size)
    else:
        train(rank, world_size)


def spawn_process(argv):
    if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
        world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        world_size = 1

    processes = []
    for rank in range(world_size):
        p = Process(target=initialize_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    app.run(spawn_process)
