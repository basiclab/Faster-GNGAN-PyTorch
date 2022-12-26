import os
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
    "SN-9L": ("SN_cifar10_cnn9", sngan.Discriminator9, dcgan.Generator),
    "GN-9L": ("GN_cifar10_cnn9", cnn.Discriminator9, dcgan.Generator),
    "SN-6L": ("SN_cifar10_cnn6", sngan.Discriminator6, dcgan.Generator),
    "GN-6L": ("GN_cifar10_cnn6", cnn.Discriminator6, dcgan.Generator),
    "SN-3L": ("SN_cifar10_cnn3", sngan.Discriminator3, dcgan.Generator),
    "GN-3L": ("GN_cifar10_cnn3", cnn.Discriminator3, dcgan.Generator),
}


def ema(data, r=0.7):
    ret = []
    for x in data:
        if len(ret) == 0:
            ret.append(x)
        else:
            ret.append(ret[-1] * (1 - r) + x * r)
    return np.array(ret)


def main():
    os.makedirs(cache_dir, exist_ok=True)

    def estimate_max_grad(D, G, loader, is_GNGAN, z_dim=128):
        D.eval()
        G.eval()
        D.requires_grad_(False)

        def calc_grad_norm(D, x):
            x.requires_grad_(True)
            if is_GNGAN:
                y, _, _ = normalize_D(D, x, wgan_loss_G)
            else:
                y = D(x)
            grad = torch.autograd.grad(y.sum(), x)[0]
            grad_norm = grad.flatten(start_dim=1).norm(dim=1)
            return grad_norm

        max_fake_gn = 0
        max_real_gn = 0
        max_midd_gn = 0
        for real, _, _ in tqdm(loader, ncols=0, desc="sampling", leave=False):
            with torch.no_grad():
                z = torch.randn((real.size(0), z_dim)).to(device)
                fake = G(z)
            max_fake_gn = max(
                max_fake_gn, calc_grad_norm(D, fake).max().cpu().item())

            real = real.to(device)
            max_real_gn = max(
                max_real_gn, calc_grad_norm(D, real).max().cpu().item())

            with torch.no_grad():
                t = torch.rand((real.size(0), 1, 1, 1)).to(device)
                midd = fake * t + real * (1 - t)
            max_midd_gn = max(
                max_midd_gn, calc_grad_norm(D, midd).max().cpu().item())

        return {
            'max_fake_gn': max_fake_gn,
            'max_real_gn': max_real_gn,
            'max_midd_gn': max_midd_gn,
        }

    dataset = torch.utils.data.Subset(
        Dataset('./data/cifar10', hflip=False, resolution=32, cr=False),
        torch.randperm(50000)[:5000])
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)

    plot_gn_data = dict()
    with tqdm(runs.items(), ncols=0, leave=False) as pbar:
        for legend, (log_name, MODEL_D, MODEL_G) in pbar:
            pbar.set_description(legend)
            log_name, MODEL_D, _ = runs[legend]
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_gn.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {legend}_{seed} from cache")
                plot_gn_data[legend] = torch.load(cache_path)
                continue

            is_GNGAN = legend.startswith('GN')
            gn_values = defaultdict(list)
            steps = []
            D = MODEL_D(resolution=32, n_classes=None).to(device)
            G = MODEL_G(resolution=32, n_classes=None, z_dim=128).to(device)
            D(torch.rand((1, 3, 32, 32)).to(device))
            for step in tqdm([1] + list(range(5000, 200001, 5000)), leave=False):
                ckpt = torch.load(os.path.join(log_path, f'{step:06}.pt'))
                G.load_state_dict(ckpt['G'])
                D.load_state_dict(ckpt['D'])
                gn_value = estimate_max_grad(D, G, loader, is_GNGAN)
                for name, value in gn_value.items():
                    gn_values[name].append(value)
                steps.append(step)
            plot_gn_data[legend] = (steps, gn_values)
            torch.save(plot_gn_data[legend], cache_path)

    ticks_fontsize = 25
    legend_fontsize = 30
    label_fontsize = 35

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [9, 10]}, sharex=True,
                                   figsize=(8, 7))
    for legend, (x, gn_values) in plot_gn_data.items():
        x = np.array(x)
        y = np.stack([
            np.array(gn_values['max_real_gn']),
            np.array(gn_values['max_fake_gn']),
            np.array(gn_values['max_midd_gn']),
        ], axis=0).max(axis=0)
        y = ema(y, 0.7)
        # xvals = np.arange(x.min(), x.max() + 1, 2000)
        # y = np.interp(xvals, x, y)
        # y = ema(y)
        # x = xvals

        if "GN" in legend:
            line_style = "-"
        else:
            line_style = "--"
        ax1.plot(x, y, line_style, label=legend, linewidth=4, alpha=0.8)
        ax2.plot(x, y, line_style, label=legend, linewidth=4, alpha=0.8)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    # with matplotlib.rc_context({'path.sketch': (3, 10, 1)}):
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)
    ax2.tick_params(bottom=True)

    # upper
    xticks = [0, 100000, 200000]
    xticks_label = ['%dk' % (x / 1000) for x in xticks]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks_label)
    ax2.tick_params(axis='x', labelsize=ticks_fontsize)
    ax2.set_xlabel('Iteration', fontsize=label_fontsize)
    ax2.set_ylim(-0.003, 0.11)  # most of the data
    ax2.set_yticks([0, 0.05, 0.1])
    ax2.tick_params(axis='y', labelsize=ticks_fontsize)
    ax2.set_ylabel(r'$\Vert\nabla_x D(x)\Vert$', fontsize=label_fontsize, y=1)

    # lower upper
    ax1.set_yticks([0.8, 0.9, 1.0])
    ax1.set_ylim(0.79, 1.01)  # outliers only
    ax1.tick_params(axis='y', labelsize=ticks_fontsize)

    ax1.legend(
        loc='lower center', fontsize=legend_fontsize,
        ncol=3, columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
        bbox_to_anchor=(0.5, 1))

    # fig.tight_layout()
    fig.subplots_adjust(
        left=0.17,
        bottom=0.17,
        right=0.97,
        top=0.78,
        wspace=0.0,
        hspace=0.15)
    fig.savefig(os.path.join(save_dir, 'vis_gn.png'))
    print("Saved to", os.path.join(save_dir, 'vis_gn.png'))


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
