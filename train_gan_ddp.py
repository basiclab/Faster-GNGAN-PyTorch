import os
import json
import math
import datetime
from shutil import copyfile

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
from absl import flags, app
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm

from source.models import gn_gan
from source.losses import HingeLoss
from source.datasets import get_dataset
from source.utils import ema, module_no_grad, set_seed
from source.optim import Adam
from metrics.score.both import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid)


net_G_models = {
    'gn-res128': gn_gan.ResGenerator128,
    'gn-res256': gn_gan.ResGenerator256,
}

net_D_models = {
    'gn-res128': gn_gan.ResDiscriminator128,
    'gn-res256': gn_gan.ResDiscriminator256,
}

datasets = [
    'celebhq_train.128.hdf5',
    'celebhq_train.128.raw',
    'celebhq.256.hdf5',
    'celebhq.256.raw',
    'lsun_church.256.hdf5',
    'lsun_church.256.raw',
    'lsun_bedroom.256.raw',
    'lsun_horse.256.raw',
]


FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'celebhq.256.hdf5', datasets, "select dataset")
flags.DEFINE_enum('arch', 'gn-res256', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps', 100000, "total number of training steps")
flags.DEFINE_integer('lr_decay_start', 100000, 'apply linearly decay to lr')
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_integer('accumulation', 1, 'gradient accumulation')
flags.DEFINE_float('G_lr', 2e-4, "Generator learning rate")
flags.DEFINE_float('D_lr', 2e-4, "Discriminator learning rate")
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('alpha', 1.0, 'alpha')
flags.DEFINE_integer('seed', 0, "random seed")
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 5000, "start step for ema")
# logging
flags.DEFINE_bool('eval_use_torch', False, 'calculate IS and FID on gpu')
flags.DEFINE_integer('eval_step', 1000, "evaluate FID and Inception Score")
flags.DEFINE_integer('save_step', 20000, "save model every this step")
flags.DEFINE_integer('num_images', 10000, '# images for evaluation')
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/GN-GAN_CELEBHQ256_RES_0', 'log folder')
flags.DEFINE_string('fid_stats', './stats/celebhq.val.256.npz', 'FID cache')
# generate
flags.DEFINE_bool('eval', False, 'load model and evaluate sample images')
flags.DEFINE_string('save', "", 'load model and save sample images to dir')
# distributed
flags.DEFINE_string('port', '55556', 'distributed port')


def image_generator(net_G):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_batch_size = FLAGS.batch_size // world_size
    with torch.no_grad():
        for idx in range(0, FLAGS.num_images, FLAGS.batch_size * 2):
            z = torch.randn(local_batch_size * 2, FLAGS.z_dim).to(rank)
            fake = (net_G(z) + 1) / 2
            fake_list = [torch.empty_like(fake) for _ in range(world_size)]
            dist.all_gather(fake_list, fake)
            fake = torch.cat(fake_list, dim=0).cpu()
            yield fake[:FLAGS.num_images - idx]
    del fake, fake_list


def eval_save(rank, world_size):
    device = torch.device('cuda:%d' % rank)

    ckpt = torch.load(
        os.path.join(FLAGS.logdir, 'best_model.pt'), map_location='cpu')
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_G)
    net_G = DDP(net_G, device_ids=[rank], output_device=rank)
    net_G.load_state_dict(ckpt['ema_G'])

    # generate fixed sample
    with torch.no_grad():
        fixed_z = torch.cat(ckpt['fixed_z'], dim=0).to(device)
        fixed_z = torch.split(fixed_z, len(fixed_z) // world_size, dim=0)
        fake = (net_G(fixed_z[rank]) + 1) / 2
        fake_list = [torch.empty_like(fake) for _ in range(world_size)]
        dist.all_gather(fake_list, fake)
        if rank == 0:
            save_image(
                torch.cat(fake_list, dim=0),
                os.path.join(FLAGS.logdir, 'fixed_sample.png'))

    # generate images for calculating IS and FID
    if rank != 0:
        for batch_images in image_generator(net_G):
            pass
    else:
        images = []
        counter = 0
        if FLAGS.save:
            os.makedirs(FLAGS.save)
        with tqdm(total=FLAGS.num_images, ncols=0,
                  desc='Sample images') as pbar:
            for batch_images in image_generator(net_G):
                images.append(batch_images)
                if FLAGS.save:
                    for image in batch_images:
                        save_image(
                            image, os.path.join(FLAGS.save, f'{counter}.png'))
                        counter += 1
                pbar.update(len(batch_images))
        images = torch.cat(images, dim=0)
        if FLAGS.eval:
            if FLAGS.save:
                (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
                    FLAGS.save, FLAGS.fid_stats, num_images=FLAGS.num_images,
                    use_torch=FLAGS.eval_use_torch, verbose=True)
            else:
                (IS, IS_std), FID = get_inception_score_and_fid(
                    images, FLAGS.fid_stats, use_torch=FLAGS.eval_use_torch,
                    verbose=True)
            print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))
    del ckpt, net_G


def evaluate(net_G):
    if dist.get_rank() != 0:
        for batch_images in image_generator(net_G):
            pass
        (IS, IS_std), FID = (None, None), None
    else:
        images = []
        with tqdm(total=FLAGS.num_images, ncols=0,
                  desc='Evaluating', leave=False) as pbar:
            for batch_images in image_generator(net_G):
                images.append(batch_images)
                pbar.update(len(batch_images))
        images = torch.cat(images, dim=0)
        (IS, IS_std), FID = get_inception_score_and_fid(
            images,
            fid_stats_path=FLAGS.fid_stats,
            use_torch=FLAGS.eval_use_torch,
            verbose=True)
        del images
    dist.barrier()
    return (IS, IS_std), FID


def infiniteloop(dataloader, sampler, step=0):
    epoch = step // len(dataloader)
    start_idx = step % len(dataloader)
    while True:
        sampler.set_epoch(epoch)
        for idx, (x, y) in enumerate(dataloader):
            if idx < start_idx:
                continue
            else:
                yield x, y
        start_idx = 0
        epoch += 1


def train(rank, world_size):
    device = torch.device('cuda:%d' % rank)

    local_batch_size = FLAGS.batch_size // world_size
    # Wait main process to create hdf5 for small dataset
    if rank == 0:
        dataset = get_dataset(FLAGS.dataset)
        dist.barrier()
    else:
        dist.barrier()
        dataset = get_dataset(FLAGS.dataset)
    dist.barrier()
    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=True, seed=FLAGS.seed, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=local_batch_size * FLAGS.accumulation * FLAGS.n_dis,
        sampler=sampler,
        num_workers=FLAGS.num_workers,
        drop_last=True)

    # model
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_G)
    net_G = DDP(net_G, device_ids=[rank], output_device=rank)
    ema_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    ema_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_G)
    ema_G = DDP(ema_G, device_ids=[rank], output_device=rank)
    net_D = net_D_models[FLAGS.arch]().to(device)
    net_D = DDP(net_D, device_ids=[rank], output_device=rank)
    net_GD = gn_gan.GenDis(net_G, net_D, alpha=FLAGS.alpha)

    # loss
    loss_fn = HingeLoss()

    # optimizer
    optim_G = Adam(net_G.parameters(), lr=FLAGS.G_lr, betas=FLAGS.betas)
    optim_D = Adam(net_D.parameters(), lr=FLAGS.D_lr, betas=FLAGS.betas)

    # scheduler
    def decay_rate(step):
        period = max(FLAGS.total_steps - FLAGS.lr_decay_start, 1)
        return 1 - max(step - FLAGS.lr_decay_start, 0) / period
    sched_G = optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=decay_rate)
    sched_D = optim.lr_scheduler.LambdaLR(optim_D, lr_lambda=decay_rate)

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
        ema_G.load_state_dict(ckpt['ema_G'])
        optim_G.load_state_dict(ckpt['optim_G'])
        optim_D.load_state_dict(ckpt['optim_D'])
        sched_G.load_state_dict(ckpt['sched_G'])
        sched_D.load_state_dict(ckpt['sched_D'])
        fixed_z = torch.cat(ckpt['fixed_z'], dim=0).to(device)
        fixed_z = torch.split(fixed_z, FLAGS.sample_size // world_size, dim=0)
        # start value
        start = ckpt['step'] + 1
        best_IS, best_FID = ckpt['best_IS'], ckpt['best_FID']
        del ckpt
    else:
        # sample fixed z
        fixed_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
        fixed_z = torch.split(fixed_z, FLAGS.sample_size // world_size, dim=0)
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

        # ema
        ema(net_G, ema_G, decay=0)

    looper = infiniteloop(dataloader, sampler, step=start - 1)
    with trange(start, FLAGS.total_steps + 1,
                desc='Training', disable=(rank != 0),
                initial=start - 1, total=FLAGS.total_steps, ncols=0) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0

            x = next(looper)[0]
            x = iter(torch.split(x, local_batch_size))
            # Discriminator
            for _ in range(FLAGS.n_dis):
                optim_D.zero_grad()
                for _ in range(FLAGS.accumulation):
                    real = next(x).to(device)
                    z = torch.randn(
                        local_batch_size, FLAGS.z_dim, device=device)
                    pred_real, pred_fake = net_GD(z, real)
                    loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)
                    loss = loss / FLAGS.accumulation
                    loss.backward()

                    loss_sum += loss.detach().item()
                    loss_real_sum += loss_real.detach().item()
                    loss_fake_sum += loss_fake.detach().item()
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
                        local_batch_size * 2, FLAGS.z_dim, device=device)
                    loss = loss_fn(net_GD(z)) / FLAGS.accumulation
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

            # sample from fixed z
            if step == 1 or step % FLAGS.sample_step == 0:
                with torch.no_grad():
                    fake_ema = (ema_G(fixed_z[rank]) + 1) / 2
                    fake_net = (net_G(fixed_z[rank]) + 1) / 2
                fake_ema_list = [
                    torch.empty_like(fake_ema) for _ in range(world_size)]
                fake_net_list = [
                    torch.empty_like(fake_net) for _ in range(world_size)]
                dist.all_gather(fake_ema_list, fake_ema)
                dist.all_gather(fake_net_list, fake_net)
                if rank == 0:
                    fake_ema = torch.cat(fake_ema_list, dim=0).cpu()
                    fake_net = torch.cat(fake_net_list, dim=0).cpu()
                    grid_ema = make_grid(fake_ema)
                    grid_net = make_grid(fake_ema)
                    writer.add_image('sample_ema', grid_ema, step)
                    writer.add_image('sample', grid_net, step)
                    save_image(
                        grid_ema,
                        os.path.join(FLAGS.logdir, 'sample', '%d.png' % step))
                del fake_ema, fake_net, fake_ema_list, fake_net_list

            # evaluate IS, FID and save latest model
            if step == 1 or step % FLAGS.eval_step == 0:
                if rank == 0:
                    ckpt = {
                        'net_G': net_G.state_dict(),
                        'net_D': net_D.state_dict(),
                        'ema_G': ema_G.state_dict(),
                        'optim_G': optim_G.state_dict(),
                        'optim_D': optim_D.state_dict(),
                        'sched_G': sched_G.state_dict(),
                        'sched_D': sched_D.state_dict(),
                        'fixed_z': fixed_z,
                        'best_IS': best_IS,
                        'best_FID': best_FID,
                        'step': step,
                    }
                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'model.pt'))
                    if step == 1 or step % FLAGS.save_step == 0:
                        torch.save(
                            ckpt, os.path.join(FLAGS.logdir, '%06d.pt' % step))

                (IS, IS_std), FID = evaluate(net_G)
                (IS_ema, IS_std_ema), FID_ema = evaluate(ema_G)

                if rank == 0:
                    if not math.isnan(FID_ema) and not math.isnan(best_FID):
                        save_as_best = (FID_ema < best_FID)
                    else:
                        save_as_best = (IS_ema > best_IS)
                    if save_as_best:
                        best_IS = IS_ema
                        best_FID = FID_ema
                        copyfile(
                            os.path.join(FLAGS.logdir, 'model.pt'),
                            os.path.join(FLAGS.logdir, 'best_model.pt'))
                    metrics = {
                        'IS': IS,
                        'IS_std': IS_std,
                        'FID': FID,
                        'IS_EMA': IS_ema,
                        'IS_std_EMA': IS_std_ema,
                        'FID_EMA': FID_ema,
                    }
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, step)
                    writer.flush()
                    path = os.path.join(FLAGS.logdir, 'eval.txt')
                    with open(path, 'a') as f:
                        metrics['step'] = step
                        f.write(json.dumps(metrics) + "\n")
                    k = len(str(FLAGS.total_steps))
                    pbar.write(
                        f"{step:{k}d}/{FLAGS.total_steps} "
                        f"IS: {IS:6.3f}({IS_std:.3f}), "
                        f"FID: {FID:.3f}, "
                        f"IS_EMA: {IS_ema:6.3f}({IS_std_ema:.3f}), "
                        f"FID_EMA: {FID_ema:.3f}")
    if rank == 0:
        writer.close()


def initialize_process(rank, world_size):
    set_seed(FLAGS.seed + rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = FLAGS.port

    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=30),
                            world_size=world_size, rank=rank)
    print("Node %d is initialized" % rank)

    if FLAGS.eval or FLAGS.save:
        eval_save(rank, world_size)
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
