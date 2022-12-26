import json
import os

import matplotlib.pyplot as plt
import numpy as np


work_dir = os.path.dirname(__file__)
save_dir = './vis/figures'
runs = {
    "DCGAN CIFAR10": {
        r"$lr_D=4\times 10^{-4}$": "run-GN_cifar10_dcgan_lrx2_0-tag-norm_nabla_fx.json",
        r"$lr_D=2\times 10^{-4}$": "run-GN_cifar10_dcgan_lrx1_0-tag-norm_nabla_fx.json",
        r"$lr_D=1\times 10^{-4}$": "run-GN_cifar10_dcgan_lrx5e-1_0-tag-norm_nabla_fx.json",
    },
    "DCGAN STL10": {
        r"$lr_D=4\times 10^{-4}$": "run-GN_stl10_dcgan_lrx2_0-tag-norm_nabla_fx.json",
        r"$lr_D=2\times 10^{-4}$": "run-GN_stl10_dcgan_lrx1_0-tag-norm_nabla_fx.json",
        r"$lr_D=1\times 10^{-4}$": "run-GN_stl10_dcgan_lrx5e-1_0-tag-norm_nabla_fx.json",
    },
    "ResNet CIFAR10": {
        r"$lr_D=4\times 10^{-4}$": "run-GN_cifar10_resnet_lrx2_0-tag-norm_nabla_fx.json",
        r"$lr_D=2\times 10^{-4}$": "run-GN_cifar10_resnet_lrx1_0-tag-norm_nabla_fx.json",
        r"$lr_D=1\times 10^{-4}$": "run-GN_cifar10_resnet_lrx1e-5_0-tag-norm_nabla_fx.json",
    },
    "ResNet STL10": {
        r"$lr_D=4\times 10^{-4}$": "run-GN_stl10_resnet_lrx2_0-tag-norm_nabla_fx.json",
        r"$lr_D=2\times 10^{-4}$": "run-GN_stl10_resnet_lrx1_0-tag-norm_nabla_fx.json",
        r"$lr_D=1\times 10^{-4}$": "run-GN_stl10_resnet_lrx5e-1_0-tag-norm_nabla_fx.json",
    },
}


if __name__ == '__main__':
    ticks_fontsize = 15
    label_fontsize = 20
    legend_fontsize = 10

    # sudo apt install texlive-latex-extra cm-super dvipng
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
    })
    styles = ["-", "--", ":"]
    with plt.style.context("fast"):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        for ax, (title, data) in zip(axs.flat, runs.items()):
            for (label, filename), linestyle in zip(data.items(), styles):
                with open(os.path.join(work_dir, filename), 'r') as f:
                    data = np.array(json.load(f))
                    x = data[:, 1]
                    y = data[:, 2]
                ax.plot(x, y, label=label, linestyle=linestyle)
            ax.legend(fontsize=legend_fontsize)
            ax.set_title(title, fontsize=label_fontsize)
            ax.set_xlabel('Iteration', fontsize=label_fontsize)
            ax.set_xticks(
                [0, 100000, 200000], ["0", "100k", "200k"],
                fontsize=ticks_fontsize)
            ax.set_ylabel(r'$\Vert\nabla_xD(x)\Vert$', fontsize=label_fontsize)
            ax.set_yscale('log')
            ax.tick_params(axis='y', labelsize=20)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'vis_nabla_fx.png'))
