import os
import glob
import multiprocessing as mp
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from vis.core import style, downsample


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs"

runs = {
    "rc10": {
        "GN-GAN": "GN_cifar10_resnet_0",
        "GN-GAN+rescale": "GN_cifar10_resnet_rescale0_xavier_0",
    },
    "bc10": {
        "GN-GAN": "GN_cifar10_biggan_0",
        "GN-GAN+rescale": "GN_cifar10_biggan_rescale0_xavier_0",
    },
    # "rs10": {
    #     "GN-GAN": "GN_stl10_resnet_0",
    #     "GN-GAN+rescale": "GN_stl10_resnet_rescale0_xavier_0",
    # },
    "rchu": {
        "GN-GAN": "GN_church256_resnet_2",
        "GN-GAN+rescale": "GN_church256_resnet_rescale0_xavier_0",
    }
}


def tf_summary_iterator(dir_path):
    import tensorflow as tf
    from tensorflow.core.util.event_pb2 import Event

    for path in glob.glob(os.path.join(dir_path, '*tfevents*')):
        try:
            for rec in tf.data.TFRecordDataset(path):
                yield Event.FromString(rec.numpy())
        except tf.errors.DataLossError:
            return


def extract_event_process(args, tag="misc/norm_nabla_fx"):
    name, legend, log_path, cache_path = args
    if os.path.exists(cache_path):
        steps, grad_norms = torch.load(cache_path)
        msg = f"Load {os.path.basename(cache_path)} from cache"
    else:
        steps = []
        grad_norms = []
        for event in tf_summary_iterator(log_path):
            for value in event.summary.value:
                if value.tag == tag:
                    steps.append(event.step)
                    grad_norms.append(value.simple_value)
        torch.save((steps, grad_norms), cache_path)
        msg = f"Extract {os.path.basename(cache_path)} from log"
    return {
        'name': name,
        'legend': legend,
        'steps': steps,
        'grad_norms': grad_norms,
        'msg': msg
    }


def main():
    os.makedirs(cache_dir, exist_ok=True)

    # Extract misc/norm_nabla_fx from logs
    args_list = []
    for name, log_names in runs.items():
        for legend, log_name in log_names.items():
            log_path = os.path.join(logdir, log_name)
            cache_path = os.path.join(cache_dir, f"{log_name}.pt")
            args_list.append((name, legend, log_path, cache_path))

    plot_gn_data = defaultdict(list)
    with mp.Pool(mp.cpu_count()) as p:
        with tqdm(p.imap(extract_event_process, args_list),
                  total=len(args_list),
                  leave=False,
                  ncols=0) as pbar:
            for results in pbar:
                plot_gn_data[results['name']].append(
                    (results['legend'], results['steps'], results['grad_norms']))
                pbar.write(results['msg'])

    # ============================= plot =============================

    for name, data in plot_gn_data.items():
        plt.figure(figsize=(8, 7))
        for legend, x, y in data:
            x, y = downsample(x, y, 100)
            plt.plot(x, y, alpha=0.8, label=legend)
        plt.xlabel('Generator Updates')

        plt.yscale('log')
        plt.ylabel(r'$\Vert\nabla_xD(x)\Vert$')
        if max(x) <= 100000:
            xticks = [0, 50000, 100000]
            yticks = range(0, 19, 6)
        elif max(x) <= 125000:
            xticks = [0, 60000, 125000]
            yticks = range(0, 9, 2)
        else:
            xticks = [0, 100000, 200000]
            yticks = range(-1, 10, 2)
        plt.xticks(
            ticks=xticks,
            labels=[f"{i / 1000:.0f}k" for i in xticks])
        plt.yticks(
            ticks=[10 ** i for i in yticks],
            labels=[r"$10^{%d}$" % i for i in yticks])

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"vis_rescaled_gn_{name}.png"))
        print("Saved to", os.path.join(save_dir, f"vis_rescaled_gn_{name}.png"))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    with style():
        main()