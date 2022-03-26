import os

import torch
from absl import app, flags
from pytorch_gan_metrics import (
    get_inception_score_and_fid_from_directory,
    get_inception_score_and_fid)
from torchvision.utils import save_image

from utils import generate_images, set_seed
from utils.args import FLAGS, net_G_models


flags.DEFINE_string('save', None, 'save samples images to directory')
flags.DEFINE_bool('ema', False, 'evaluate ema model')


def main(argv):
    device = torch.device('cuda:0')

    ckpt = torch.load(
        os.path.join(FLAGS.logdir, 'best_model.pt'), map_location='cpu')
    net_G = net_G_models[FLAGS.model](FLAGS.z_dim).to(device)
    if FLAGS.ema:
        net_G.load_state_dict(ckpt['ema_G'])
    else:
        net_G.load_state_dict(ckpt['net_G'])

    # generate random images
    if FLAGS.save:
        os.makedirs(FLAGS.save)
        counter = 0
    else:
        images = []
    for batch_images in generate_images(net_G,
                                        FLAGS.batch_size_G,
                                        FLAGS.num_images,
                                        FLAGS.z_dim,
                                        FLAGS.n_classes,
                                        verbose=True):
        if FLAGS.save:
            for image in batch_images:
                save_image(image, os.path.join(FLAGS.save, f'{counter}.png'))
                counter += 1
        else:
            images.append(batch_images.cpu())
    if FLAGS.save:
        (IS, IS_std), FID = get_inception_score_and_fid_from_directory(
            FLAGS.save, FLAGS.fid_stats, verbose=True)
    else:
        (IS, IS_std), FID = get_inception_score_and_fid(
            torch.cat(images, dim=0), FLAGS.fid_stats, verbose=True)
    print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))


if __name__ == '__main__':
    app.run(main)
