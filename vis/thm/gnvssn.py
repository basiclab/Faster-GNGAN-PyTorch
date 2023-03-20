import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import sngan, cnn, dcgan
from training.datasets import Dataset
from vis.core import style, ema, calc_grad_norm


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


@torch.no_grad()
def estimate_max_gn(D, G, loader, is_GNGAN, z_dim=128):
    G.eval()
    D.eval()
    D.requires_grad_(False)

    max_grad_norm = 0
    with tqdm(loader, ncols=0, leave=False) as pbar:
        for real, _, _ in pbar:
            real = real.to(device)
            fake = G(torch.randn(real.size(0), z_dim, device=device))
            midd = torch.lerp(real, fake, torch.rand(real.size(0), 1, 1, 1, device=device))

            # real data points
            init_gn, last_fn, max_gn = calc_grad_norm(D, real, is_GNGAN, step=1)
            max_grad_norm = max(max_grad_norm, max_gn)
            # pbar.write(f"{init_gn:.4f} -> {last_fn:.4f}: max is {max_gn:.4f}")
            # fake data points
            init_gn, last_fn, max_gn = calc_grad_norm(D, fake, is_GNGAN, step=1)
            max_grad_norm = max(max_grad_norm, max_gn)
            # interpolation points
            init_gn, last_fn, max_gn = calc_grad_norm(D, midd, is_GNGAN, step=1)
            max_grad_norm = max(max_grad_norm, max_gn)

            pbar.set_postfix(max_grad_norm=max_grad_norm)

    return max_grad_norm


def main():
    os.makedirs(cache_dir, exist_ok=True)

    indices = torch.randperm(
        50000, generator=torch.Generator().manual_seed(0))[:1000]
    dataset = torch.utils.data.Subset(
        Dataset('./data/cifar10', hflip=False, resolution=32, cr=False),
        indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)

    plot_data = dict()
    with tqdm(runs.items(), ncols=0, leave=False) as pbar:
        for legend, (log_name, MODEL_D, MODEL_G) in pbar:
            pbar.set_description(legend)
            log_name, MODEL_D, _ = runs[legend]
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_gnvssn.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {legend}_{seed} from cache")
                plot_data[legend] = torch.load(cache_path)
                continue

            is_GNGAN = legend.startswith('GN')
            D = MODEL_D(resolution=32, n_classes=None).to(device)
            G = MODEL_G(resolution=32, n_classes=None, z_dim=128).to(device)
            D(torch.rand((1, 3, 32, 32)).to(device))
            max_grad_norms = []
            steps = []
            for step in tqdm([1] + list(range(5000, 200001, 5000)), leave=False):
                ckpt = torch.load(os.path.join(log_path, f'{step:06}.pt'))
                G.load_state_dict(ckpt['G'])
                D.load_state_dict(ckpt['D'])
                max_grad_norm = estimate_max_gn(D, G, loader, is_GNGAN)
                max_grad_norms.append(max_grad_norm)
                steps.append(step)
            plot_data[legend] = (steps, max_grad_norms)
            torch.save(plot_data[legend], cache_path)

    # ============================= plot =============================

    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [9, 10]}, sharex=True, figsize=(8, 6))
    for legend, (x, max_grad_norms) in plot_data.items():
        x = np.array(x)
        y = np.array(max_grad_norms)
        y = ema(y, 0.8)

        if "GN" in legend:
            line_style = "-"
        else:
            line_style = "--"
        ax1.plot(x, y, line_style, label=legend, linewidth=2, alpha=0.8)
        ax2.plot(x, y, line_style, label=legend, linewidth=2, alpha=0.8)

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
    ax2.set_xlabel('Update Iteration (Generator)')
    yticks = [0, 0.05, 0.1]
    ax2.set_yticks(yticks)
    ax2.set_ylim(-0.003, 0.11)
    ax2.set_ylabel(r'$\max_x\Vert\nabla_x\hat{D}(x)\Vert$', y=1)

    # lower upper
    yticks = [0.7, 0.8, 0.9, 1.0]
    ax1.set_yticks(yticks)
    ax1.set_ylim(0.69, 1.01)

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

    fig.subplots_adjust(
        left=0.18,
        bottom=0.17,
        right=0.97,
        top=0.73,
        wspace=0.0,
        hspace=0.1)
    fig.savefig(os.path.join(save_dir, 'vis_gnvssn.png'))
    print("Saved to", os.path.join(save_dir, 'vis_gnvssn.png'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    with style():
        main()
