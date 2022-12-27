import os
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from training.models import sngan, dcgan
from training.datasets import Dataset
from .ema import ema


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs/Ablation"
runs = {
    "SN-9L": ("SN_cifar10_cnn9", sngan.Discriminator9, dcgan.Generator),
    # "SN-6L": ("SN_cifar10_cnn6", sngan.Discriminator6, dcgan.Generator),
    # "SN-3L": ("SN_cifar10_cnn3", sngan.Discriminator3, dcgan.Generator),
}


class Hook:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.output1 = None
        self.output2 = None

    def hook_fn(self, module, input, output):
        self.output1 = self.output2
        self.output2 = output


@torch.no_grad()
def calc_max_slop(D, x, hooks, eps=1e-5):
    dx = (torch.rand_like(x) * 2 - 1) * eps
    x1 = x
    x2 = (x + dx).clamp(-1, 1)
    D(x)
    D(x + dx)
    slops = []
    for hook in hooks:
        y1 = hook.output1
        y2 = hook.output2
        dy = (y2 - y1).flatten(start_dim=1)
        dx = (x2 - x1).flatten(start_dim=1)
        slop = torch.linalg.norm(dy, dim=1) / (
            torch.linalg.norm(dx, dim=1) + 1e-10)
        if torch.any(slop.gt(1)):
            i = torch.argmax(slop)
            torch.save(x[i], "x.pt")
            torch.save(dx[i], "dx.pt")
        slops.append(slop.max().cpu().item())
    return np.array(slops)


@torch.no_grad()
def estimate_lipschitz(D, G, loader, hooks, z_dim=128):
    D.eval()
    G.eval()

    lipschitz = np.zeros(len(hooks))
    for real, _, _ in tqdm(loader, ncols=0, desc="sampling", leave=False):
        real = real.to(device)
        fake = G(torch.randn(real.size(0), z_dim, device=device))
        t = torch.rand(real.size(0), 1, 1, 1, device=device)
        midd = fake * t + real * (1 - t)

        # fake data points
        lipschitz = np.maximum(lipschitz, calc_max_slop(D, fake, hooks))
        # real data points
        lipschitz = np.maximum(lipschitz, calc_max_slop(D, real, hooks))
        # interpolation points
        lipschitz = np.maximum(lipschitz, calc_max_slop(D, midd, hooks))

        # perturbed fake/real data points
        for _ in range(10):
            noise = (torch.rand_like(real) * 2 - 1) * 0.1
            lipschitz = np.maximum(
                lipschitz, calc_max_slop(D, fake + noise, hooks))
            lipschitz = np.maximum(
                lipschitz, calc_max_slop(D, real + noise, hooks))

    for _ in range(100):
        noise = torch.rand_like(real) * 2 - 1
        lipschitz = np.maximum(lipschitz, calc_max_slop(D, noise, hooks))

    # steps = (np.arange(len(hooks)) + 1) / len(hooks)
    steps = np.arange(len(hooks)) + 1

    return steps, lipschitz


def main():
    os.makedirs(cache_dir, exist_ok=True)

    dataset = torch.utils.data.Subset(
        Dataset('./data/cifar10', hflip=False, resolution=32, cr=False),
        torch.randperm(50000)[:5000])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)

    plot_lipschitz_data = dict()
    with tqdm(runs.items(), ncols=0, leave=False) as pbar:
        for legend, (log_name, MODEL_D, MODEL_G) in pbar:
            pbar.set_description(legend)
            log_name, MODEL_D, _ = runs[legend]
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_lip.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {legend}_{seed} from cache")
                plot_lipschitz_data[legend] = torch.load(cache_path)
                continue

            D = MODEL_D(resolution=32, n_classes=None).to(device)
            G = MODEL_G(resolution=32, n_classes=None, z_dim=128).to(device)
            D(torch.rand((1, 3, 32, 32)).to(device))
            ckpt = torch.load(os.path.join(log_path, '200000.pt'))
            G.load_state_dict(ckpt['G'])
            D.load_state_dict(ckpt['D'])

            hooks = []
            for module in D.modules():
                if isinstance(module, sngan.SpectralNorm):
                    hooks.append(Hook(module))

            steps, lipschitz = estimate_lipschitz(D, G, loader, hooks)
            plot_lipschitz_data[legend] = (steps, lipschitz)
            pbar.write(", ".join(f"{x:.3f}" for x in lipschitz))
            # torch.save(plot_lipschitz_data[legend], cache_path)

            for hook in hooks:
                hook.handle.remove()

    # ============================= plot =============================

    ticks_fontsize = 25
    legend_fontsize = 30
    label_fontsize = 35

    plt.figure(figsize=(8, 7))
    for legend, (x, lipschitz) in plot_lipschitz_data.items():
        x = np.array(x)
        y = np.array(lipschitz)

        if "GN" in legend:
            line_style = "-"
        else:
            line_style = "--"
        plt.plot(x, y, line_style, label=legend, linewidth=4, alpha=0.8)

    xticks = [1, 5, 10]
    plt.xticks(xticks, fontsize=ticks_fontsize)
    plt.xlabel('$k$', fontsize=label_fontsize)
    yticks = [0, 0.3]
    plt.yticks(yticks, fontsize=ticks_fontsize)
    plt.ylabel(r'$L_{f_k(x)}$', fontsize=label_fontsize)

    plt.legend(
        loc='lower center', fontsize=legend_fontsize,
        ncol=3, columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
        bbox_to_anchor=(0.5, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_decay_lipschitz.png'))
    print("Saved to", os.path.join(save_dir, 'vis_decay_lipschitz.png'))


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
