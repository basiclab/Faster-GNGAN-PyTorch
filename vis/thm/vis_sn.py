import os
from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from training.models import sngan, cnn, dcgan


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
colors = {
    legend: f"C{i}" for i, legend in enumerate(runs.keys())
}


def main():
    os.makedirs(cache_dir, exist_ok=True)

    # Estimate spectral norm of each layer
    class ShapeHook:
        def __init__(self, module):
            self.module = module
            self.input_shape = None

        def __call__(self, module, inputs, outputs):
            self.module = module
            self.input_shape = inputs[0].shape

        @staticmethod
        def register(module):
            hook = ShapeHook(module)
            handle = module.register_forward_hook(hook)
            hook.handle = handle
            return hook

    vis_sn_data = defaultdict(list)

    whitelist = ['SN-9L', 'GN-9L']
    with tqdm(whitelist, ncols=0, leave=False) as pbar:
        for legend in pbar:
            pbar.set_description(legend)
            log_name, MODEL_D, _ = runs[legend]
            for seed in range(5):
                log_path = os.path.join(logdir, f"{log_name}_{seed}")
                cache_path = os.path.join(cache_dir, f"{legend}_{seed}_sn.pt")
                if os.path.exists(log_path):
                    break

            if os.path.exists(cache_path):
                pbar.write(f"Load {legend}_{seed} from cache")
                vis_sn_data[legend] = torch.load(cache_path)
                continue

            D = MODEL_D(resolution=32, n_classes=None).to(device)
            D.eval()

            # Get the input shape of each layer
            hooks = []
            for module in D.modules():
                if legend.startswith("SN"):
                    if not isinstance(module, sngan.SpectralNorm):
                        continue
                    if not isinstance(module.module, nn.Conv2d):
                        continue
                else:
                    if not isinstance(module, nn.Conv2d):
                        continue
                hook = ShapeHook.register(module)
                hooks.append(hook)

            # initialize shape of u and v
            D(torch.rand((1, 3, 32, 32)).to(device))
            ckpt = torch.load(os.path.join(log_path, 'model.pt'))
            D.load_state_dict(ckpt['D'])

            for hook in hooks:
                sn = sngan.auto_spectral_norm(hook.module, hook.input_shape)
                vis_sn_data[legend].append(sn.cpu().item())
            torch.save(vis_sn_data[legend], cache_path)

    # Plot spectral norm
    ticks_fontsize = 25
    legend_fontsize = 30
    label_fontsize = 35
    bar_width = 0.4

    plt.figure(figsize=(8, 7))
    for i, (legend, sn) in enumerate(vis_sn_data.items()):
        print(legend, ", ".join(f"{v:.3f}" for v in sn))
        if legend.startswith('SN'):
            enhence = 1
        else:
            enhence = 1
        r = np.arange(len(sn)) + i * bar_width
        bar = plt.bar(r, np.array(sn) * enhence, width=bar_width,
                      edgecolor='white', label=legend, alpha=0.8)
        for rect in bar:
            height = rect.get_height()
            if height > 30:
                plt.text(rect.get_x() + rect.get_width() / 2.0, 30,
                         f'{height:.0f}', ha='center', va='bottom',
                         fontsize=ticks_fontsize)

    tick_center = np.arange(len(sn)) + (len(vis_sn_data) - 1) * 0.5 * bar_width
    xtick_labels = np.arange(len(sn)) + 1
    plt.xticks(tick_center, xtick_labels, fontsize=ticks_fontsize)
    plt.xlabel('$k$-th CNN Layer', fontsize=label_fontsize)

    yticks = [1, 10, 20, 30]
    plt.yticks(yticks, yticks, fontsize=ticks_fontsize)
    plt.ylabel(r'$L_{\mathbf{W}_k}$', fontsize=label_fontsize)
    plt.ylim(0, 30)
    plt.axhline(y=1, color='black', linestyle='dotted', alpha=0.5)

    plt.legend(
        loc='lower center', fontsize=legend_fontsize,
        ncol=3, columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
        bbox_to_anchor=(0.5, 1.05)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vis_sn.png'))
    print("Saved to", os.path.join(save_dir, 'vis_sn.png'))


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
