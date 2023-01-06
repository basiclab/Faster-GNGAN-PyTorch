import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import base, dcgan
from training.datasets import Dataset
from vis.core import style


device = torch.device('cuda:0')
work_dir = os.path.dirname(__file__)
cache_dir = os.path.join(work_dir, "cache")
save_dir = './vis/figures'
logdir = "./logs"
runs = {
    "GN-GAN": ("GN_cifar10_dcgan", dcgan.Discriminator, dcgan.Generator),
    "GN-GAN + rescale": ("GN_cifar10_dcgan_rescale0", dcgan.Discriminator, dcgan.Generator),
}
markers = ['o', 'D']


class Hook:
    def __init__(self, module):
        self.handle_fw = module.register_forward_hook(self.hook_fw)
        # self.handle_bw = module.register_full_backward_hook(self.hook_bw)
        self.norm_fw_mean = None
        self.norm_fw_std = None
        self.norm_bw_mean = None
        self.norm_bw_std = None

    @torch.no_grad()
    def hook_fw(self, module, input, output):
        output.register_hook(self.hook_bw)
        norm = torch.linalg.vector_norm(output.flatten(start_dim=1), dim=1)
        self.norm_fw_mean = norm.mean().item()
        self.norm_fw_std = norm.std().item()

    @torch.no_grad()
    def hook_bw(self, grad):
        norm = torch.linalg.vector_norm(grad.flatten(start_dim=1), dim=1)
        self.norm_bw_mean = norm.mean().item()
        self.norm_bw_std = norm.std().item()

    def remove(self):
        self.handle_fw.remove()
        self.handle_bw.remove()


@torch.no_grad()
def estimate_max_gn(D, G, loader, z_dim=128):
    G.eval()
    D.eval()
    D.requires_grad_(False)

    hooks = []
    for module in D.modules():
        if isinstance(module, base.RescalableWrapper):
            if isinstance(module.module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(Hook(module))
                print(len(hooks), module.module)

    real = next(iter(loader))[0].to(device)
    fake = G(torch.randn(real.size(0), z_dim, device=device))
    with torch.enable_grad():
        x = torch.cat([real, fake], dim=0)
        x.requires_grad_(True)
        y = D(x)
        y.sum().backward()

    gn_means_fw = []
    gn_stds_fw = []
    gn_means_bw = []
    gn_stds_bw = []
    steps = []
    for i, hook in enumerate(hooks):
        gn_means_fw.append(hook.norm_fw_mean)
        gn_stds_fw.append(hook.norm_fw_std)
        gn_means_bw.append(hook.norm_bw_mean)
        gn_stds_bw.append(hook.norm_bw_std)
        steps.append(i)

    return steps, gn_means_fw, gn_stds_fw, gn_means_bw, gn_stds_bw


def main():
    os.makedirs(cache_dir, exist_ok=True)

    indices = torch.randperm(
        50000, generator=torch.Generator().manual_seed(0))[:5000]
    dataset = torch.utils.data.Subset(
        Dataset('./data/cifar10', hflip=False, resolution=32, cr=False),
        indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=4, shuffle=False)

    plot_fw_data = dict()
    plot_bw_data = dict()
    with tqdm(runs.items(), ncols=0, leave=False) as pbar:
        for legend, (log_name, MODEL_D, MODEL_G) in pbar:
            pbar.set_description(legend)
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{log_name}_{seed}.propagation.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {log_name}_{seed} from cache")
                plot_fw_data[legend], plot_bw_data[legend] = torch.load(cache_path)
                continue

            D = MODEL_D(resolution=32, n_classes=None).to(device)
            G = MODEL_G(resolution=32, n_classes=None, z_dim=128).to(device)
            D(torch.rand((1, 3, 32, 32)).to(device))
            ckpt = torch.load(os.path.join(log_path, '200000.pt'))
            G.load_state_dict(ckpt['G'])
            D.load_state_dict(ckpt['D'])

            steps, gn_means_fw, gn_stds_fw, gn_means_bw, gn_stds_bw = \
                estimate_max_gn(D, G, loader)
            plot_fw_data[legend] = (steps, gn_means_fw, gn_stds_fw)
            plot_bw_data[legend] = (steps, gn_means_bw, gn_stds_bw)
            pbar.write(", ".join(f"{x:.3f}" for x in gn_means_fw))
            torch.save((plot_fw_data[legend], plot_bw_data[legend]), cache_path)

    # ============================= plot =============================

    ticks_fontsize = 35
    legend_fontsize = 22
    label_fontsize = 40
    markersize = 10

    plt.figure("fw", figsize=(8, 7))
    lines = []
    legends = []
    for marker, (legend, (x, y, _)) in zip(markers, plot_fw_data.items()):
        x = np.array(x) + 1
        y = np.array(y)

        marker, = plt.plot(x, y, linestyle='', marker=marker)
        line2d, = plt.plot(x, y, alpha=0.8, color=marker.get_color())

        lines.append((marker, line2d))
        legends.append(legend)

    plt.xticks(ticks=x, labels=x)
    plt.xlabel('$k$')

    yticks = [1, 2, 3, 4, 5, 6]
    plt.yscale('log')
    plt.yticks(
        ticks=[10 ** i for i in yticks],
        labels=[f"$10^{{{i}}}$" for i in yticks])
    plt.ylim(10 ** 1, 10 ** 6)
    plt.ylabel(r'$\Vert f_k(x)\Vert$')
    plt.grid(axis='x')

    plt.legend(
        lines, legends, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_forward.png'))
    print("Saved to", os.path.join(save_dir, 'vis_forward.png'))

    # ============================= plot =============================

    plt.figure("bw", figsize=(8, 7))
    lines = []
    legends = []
    for marker, (legend, (x, y, _)) in zip(markers, plot_bw_data.items()):
        x = np.array(x) + 1
        y = np.array(y)

        marker, = plt.plot(x, y, linestyle='', marker=marker)
        line2d, = plt.plot(x, y, alpha=0.8, color=marker.get_color())

        lines.append((marker, line2d))
        legends.append(legend)

    plt.xticks(ticks=x, labels=x)
    plt.xlabel('$k$')

    yticks = [0, 1, 2, 3, 4, 5, 6]
    plt.yscale('log')
    plt.yticks(
        ticks=[10 ** i for i in yticks],
        labels=[f"$10^{{{i}}}$" for i in yticks])
    plt.ylim(10 ** -0.5, 10 ** 6)
    plt.ylabel(r'$\Vert\nabla_{f_k(x)}D(x)\Vert$')
    plt.grid(axis='x')

    plt.legend(
        lines, legends, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_backward.png'))
    print("Saved to", os.path.join(save_dir, 'vis_backward.png'))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

    with style():
        main()
