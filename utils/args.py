from absl import flags

from models import biggan, dcgan, resnet
from .losses import BCEWithLogits, HingeLoss, Wasserstein


net_G_models = {
    'resnet.128': resnet.ResGenerator128,
    'resnet.256': resnet.ResGenerator256,
    'dcgan.32': dcgan.Generator32,
    'dcgan.48': dcgan.Generator48,
    'resnet.32': resnet.ResGenerator32,
    'resnet.48': resnet.ResGenerator48,
    'biggan.32': biggan.Generator32,
    'biggan.128': biggan.Generator128,
}

net_D_models = {
    'resnet.128': resnet.ResDiscriminator128,
    'resnet.256': resnet.ResDiscriminator256,
    'dcgan.32': dcgan.Discriminator32,
    'dcgan.48': dcgan.Discriminator48,
    'resnet.32': resnet.ResDiscriminator32,
    'resnet.48': resnet.ResDiscriminator48,
    'biggan.32': biggan.Discriminator32,
    'biggan.128': biggan.Discriminator128,
}

datasets = [
    'cifar10.32',
    'stl10.48',
    'celebahq.128',
    'celebahq.256',
    'lsun_church.256',
    'lsun_bedroom.256',
    'lsun_horse.256',
    'imagenet.128'
]

loss_fns = {
    'hinge': HingeLoss,
    'bce': BCEWithLogits,
    'wass': Wasserstein,
}

FLAGS = flags.FLAGS
# resume
flags.DEFINE_bool('resume', False, 'resume from logdir')
# model and training
flags.DEFINE_enum('dataset', 'cifar10.32', datasets, "select dataset")
flags.DEFINE_enum('model', 'resnet.32', net_G_models.keys(), "architecture")
flags.DEFINE_enum('loss', 'hinge', loss_fns.keys(), "loss function")
flags.DEFINE_integer('total_steps', 200000, "total number of training steps")
flags.DEFINE_integer('batch_size_D', 64, "batch size")
flags.DEFINE_integer('batch_size_G', 128, "batch size")
flags.DEFINE_integer('accumulation', 1, 'gradient accumulation')
flags.DEFINE_integer('num_workers', 8, "dataloader workers")
flags.DEFINE_float('lr_D', 4e-4, "Discriminator learning rate")
flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_integer('lr_decay_start', 0, 'apply linearly decay to lr')
flags.DEFINE_multi_float('betas', [0.0, 0.9], "for Adam")
flags.DEFINE_integer('n_dis', 5, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_bool('rescale', False, 'rescale output of each layer')
flags.DEFINE_float('alpha', 1.0, 'hyper parameter for rescaling')
flags.DEFINE_integer('seed', 0, "random seed")
# other regularization
flags.DEFINE_float('cr', 0, "lambda for consistency regularization")
flags.DEFINE_float('gp', 0, "lambda for gradient penalty")
# conditional
flags.DEFINE_integer('n_classes', 1, 'the number of classes in dataset')
# ema
flags.DEFINE_float('ema_decay', 0.9999, "ema decay rate")
flags.DEFINE_integer('ema_start', 0, "start step for ema")
# logging
flags.DEFINE_integer('sample_step', 500, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('eval_step', 5000, "evaluate FID and Inception Score")
flags.DEFINE_integer('save_step', 20000, "save model every this step")
flags.DEFINE_integer('num_images', 50000, '# images for evaluation')
flags.DEFINE_string('fid_stats', './stats/cifar10.train.npz', 'FID statistics')
flags.DEFINE_string('logdir', './logs/GAN_CIFAR10_RES_0', 'log folder')
# distributed
flags.DEFINE_string('port', '56789', 'inter-process communication port')
