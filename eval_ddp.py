import datetime
import os

import torch
import torch.distributed as dist
from absl import app, flags
from pytorch_gan_metrics import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
from torchvision.utils import save_image

from utils import generate_images_ddp, set_seed
from utils.args import FLAGS, net_G_models


flags.DEFINE_string('save', None, 'save samples images to directory')
flags.DEFINE_bool('ema', True, 'evaluate ema model')


def main(rank, world_size):
    device = torch.device('cuda:%d' % rank)

    ckpt = torch.load(
        os.path.join(FLAGS.logdir, 'best_model.pt'), map_location='cpu')
    net_G = net_G_models[FLAGS.model](FLAGS.z_dim).to(device)
    net_G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_G)
    if FLAGS.ema:
        net_G.load_state_dict(ckpt['ema_G'])
    else:
        net_G.load_state_dict(ckpt['net_G'])
    net_G = DDP(net_G, device_ids=[rank], output_device=rank)

    # generate random images
    if rank == 0:
        if FLAGS.save:
            os.makedirs(FLAGS.save)
            counter = 0
        else:
            images = []
    for batch_images in generate_images_ddp(net_G,
                                            FLAGS.batch_size_G,
                                            FLAGS.num_images,
                                            FLAGS.z_dim,
                                            FLAGS.n_classes,
                                            verbose=True):
        if rank != 0:
            continue
        if FLAGS.save:
            for image in batch_images:
                save_image(image, os.path.join(FLAGS.save, f'{counter}.png'))
                counter += 1
        else:
            images.append(batch_images.cpu())
    if rank == 0:
        if FLAGS.save:
            (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
                FLAGS.save, FLAGS.fid_stats, verbose=True)
        else:
            (IS, IS_std), FID = get_inception_score_and_fid(
                torch.cat(images, dim=0), FLAGS.fid_stats, verbose=True)
        print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))


def initialize_process(rank, world_size):
    set_seed(FLAGS.seed + rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = FLAGS.port
    dist.init_process_group(
        'nccl', timeout=datetime.timedelta(seconds=30), world_size=world_size,
        rank=rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    print("Node %d is initialized" % rank)
    main(rank, world_size)


def spawn_process(_):
    world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(','))

    processes = []
    for rank in range(world_size):
        p = Process(target=initialize_process, args=(rank, world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    app.run(spawn_process)
