import os
import glob
import multiprocessing as mp
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import sngan, cnn, dcgan
from training.datasets import Dataset
from training.losses import wgan_loss_G
from training.gn import normalize_D


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs/Ablation"
runs = {
    "SN-9L": "SN_cifar10_cnn9",
    "GN-9L": "GN_cifar10_cnn9",
    "SN-6L": "SN_cifar10_cnn6",
    "GN-6L": "GN_cifar10_cnn6",
    "SN-3L": "SN_cifar10_cnn3",
    "GN-3L": "GN_cifar10_cnn3",
}


def extract_event_process(args):
    def tf_summary_iterator(dir_path):
        import tensorflow as tf
        from tensorflow.core.util.event_pb2 import Event

        for path in glob.glob(os.path.join(dir_path, '*tfevents*')):
            try:
                for rec in tf.data.TFRecordDataset(path):
                    yield Event.FromString(rec.numpy())
            except tf.errors.DataLossError:
                return

    legend, log_name, seed = args
    log_path = os.path.join(logdir, f"{log_name}_{seed}")
    cache_path = os.path.join(cache_dir, f"{legend}_{seed}_fid.pt")
    if os.path.exists(cache_path):
        x_fid, y_fid = torch.load(cache_path)
        msg = f"Load {legend}_{seed} from cache"
        torch.save((x_fid, y_fid), cache_path)
    else:
        x_fid = []
        y_fid = []
        for event in tf_summary_iterator(log_path):
            for value in event.summary.value:
                if value.tag == "FID":
                    x_fid.append(event.step)
                    y_fid.append(value.simple_value)
        torch.save((x_fid, y_fid), cache_path)
        msg = f"Extract {legend}_{seed} from log"
    return {
        'legend': legend,
        'x_fid': x_fid,
        'y_fid': y_fid,
        'msg': msg
    }


def main():
    os.makedirs(cache_dir, exist_ok=True)

    # Extract norm_nabla_hatfx and FID from logs
    args_list = []
    for legend, log_name in runs.items():
        for seed in range(5):
            log_path = os.path.join(logdir, f"{log_name}_{seed}")
            if os.path.exists(log_path):
                args_list.append((legend, log_name, seed))

    vis_fid_data = defaultdict(list)
    with mp.Pool(6) as p:
        with tqdm(p.imap_unordered(extract_event_process, args_list),
                  total=len(args_list),
                  leave=False,
                  ncols=0) as pbar:
            for results in pbar:
                vis_fid_data[results['legend']].append((results['x_fid'], results['y_fid']))
                pbar.write(results['msg'])

    ticks_fontsize = 25
    legend_fontsize = 30
    label_fontsize = 40

    plt.figure(figsize=(8, 7))
    for legend, steps_fids_list in vis_fid_data.items():
        avg_fids = defaultdict(list)
        for steps, fids in steps_fids_list:
            assert len(steps) == len(fids)
            # fids = ema(fids)
            for step, fid in zip(steps, fids):
                avg_fids[step].append(fid)
        x = []
        y = []
        sigma = []
        for step in sorted(avg_fids.keys()):
            if len(avg_fids[step]) == len(steps_fids_list):
                x.append(step)
                y.append(np.mean(avg_fids[step]))
                sigma.append(np.std(avg_fids[step]))
        x = np.array(x)
        y = np.array(y)
        sigma = np.array(sigma)
        if "GN" in legend:
            line_style = "-"
        else:
            line_style = "--"
        plt.plot(x, y, line_style, label=legend, linewidth=3, alpha=0.8)
        plt.fill_between(x, y + sigma, y - sigma, alpha=0.5)

    xticks = [0, 100000, 200000]
    xticks_label = ['%dk' % (x / 1000) for x in xticks]
    plt.xticks(xticks, xticks_label, fontsize=ticks_fontsize)
    plt.xlabel('Iteration', fontsize=label_fontsize)
    # ax.tick_params(axis='x', labelsize=label_fontsize)

    yticks = [0, 30, 60, 90]
    plt.yticks(yticks, fontsize=ticks_fontsize)
    plt.ylim(10, 110)
    plt.ylabel('FID', fontsize=label_fontsize, y=0.54)

    plt.legend(
        loc='lower center', fontsize=legend_fontsize,
        ncol=3, columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
        bbox_to_anchor=(0.5, 1.0)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_fid.png'))
    print("Saved to", os.path.join(save_dir, 'vis_fid.png'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    # sudo apt install texlive-latex-extra cm-super dvipng
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })
    with plt.style.context("fast"):
        main()
