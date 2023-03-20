import os
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import dcgan
from training.datasets import Dataset
from vis.core import style, calc_grad_norm


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs"
runs = {
    "GP-DCGAN": ("GP_cifar10_dcgan", dcgan.Discriminator, dcgan.Generator),
    "GN-DCGAN": ("GN_cifar10_dcgan", dcgan.Discriminator, dcgan.Generator),
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
            fake = G(torch.randn((real.size(0), z_dim)).to(device))
            t = torch.rand((real.size(0), 1, 1, 1)).to(device)
            midd = fake * t + real * (1 - t)

            # real data points
            init_gn, last_fn, max_gn = calc_grad_norm(D, real, is_GNGAN)
            max_grad_norm = max(max_grad_norm, max_gn)
            pbar.write(f"{init_gn:.4f} -> {last_fn:.4f}: max is {max_gn:.4f}")
            # fake data points
            init_gn, last_fn, max_gn = calc_grad_norm(D, fake, is_GNGAN)
            max_grad_norm = max(max_grad_norm, max_gn)
            # interpolation points
            init_gn, last_fn, max_gn = calc_grad_norm(D, midd, is_GNGAN)
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
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_gnvsgp.pt")
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

    for legend, (x, max_grad_norms) in plot_data.items():
        x = np.array(x)
        y = np.array(max_grad_norms)
        plt.plot(x, y, alpha=0.8, label=legend)

    xticks = [0, 100000, 200000]
    xticks_label = ['%dk' % (x / 1000) for x in xticks]
    plt.xticks(xticks, xticks_label)
    plt.xlabel('Update Iteration (Generator)')

    yticks = [0.5, 1.0, 1.5]
    plt.ylim(0.5, 1.5)
    plt.yticks(yticks)
    plt.ylabel(r'$\max_x\Vert\nabla_x\hat{D}(x)\Vert$')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_gnvsgp.png'))
    print("Saved to", os.path.join(save_dir, 'vis_gnvsgp.png'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
    with style():
        main()
