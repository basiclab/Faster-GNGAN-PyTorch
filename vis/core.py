import os
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from training.gn import normalize_D
from training.losses import wgan_loss_G


def ema(data, r=0.7):
    ret = []
    for x in data:
        if len(ret) == 0:
            ret.append(x)
        else:
            ret.append(ret[-1] * (1 - r) + x * r)
    return np.array(ret)


def downsample(x, y, num_samples=1000):
    if len(x) <= num_samples:
        return x, y
    else:
        return x[::len(x) // num_samples], y[::len(x) // num_samples]


@torch.enable_grad()
def calc_grad_norm(D, x, is_GNGAN=False, step=20, lr=1000):
    init_gn = 0
    max_gn = 0
    for i in range(step):
        x.requires_grad_(True)
        if is_GNGAN:
            y, _, _ = normalize_D(D, x, wgan_loss_G)
        else:
            y = D(x)
        y = torch.linalg.vector_norm(y.flatten(start_dim=1), dim=1)
        grad = torch.autograd.grad(
            y.sum(), x, create_graph=True, retain_graph=True)[0]
        grad_norm = grad.flatten(start_dim=1).norm(dim=1)
        if i == 0:
            init_gn = grad_norm[0].detach().cpu().numpy()
        grad_grad = torch.autograd.grad(grad_norm.sum(), x)[0]
        with torch.no_grad():
            x = x + lr * grad_grad
            max_gn = max(max_gn, grad_norm.detach().max().item())
    last_fn = grad_norm[0].detach().cpu().numpy()
    return init_gn, last_fn, max_gn


@contextmanager
def style(legend_fontsize=30, label_fontsize=40, ticks_labelsize=25):
    plt.rcParams.update({
        # Latex font
        # sudo apt install texlive-latex-extra cm-super dvipng
        "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',

        # figsize
        'figure.figsize': (8, 7),

        # grid color and border color
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.color': '#808080',
        'grid.linewidth': 2,
        'axes.edgecolor': '#80808040',
        'axes.linewidth': 2,

        # legend style
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.3,
        'legend.columnspacing': 0.7,
        'legend.fontsize': legend_fontsize,

        # xy label and tick size
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': ticks_labelsize,
        'ytick.labelsize': ticks_labelsize,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,

        # line and marker style
        'lines.linewidth': 3,
        'lines.markersize': 10,
    })

    with plt.style.context("fast"):
        yield

    matplotlib.rcdefaults()
