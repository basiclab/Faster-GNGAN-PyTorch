import time

import torch
import torch.nn as nn
from tqdm import trange

from training.gn import normalize_D
from training.models import dcgan, resnet


def si_format(v, unit=""):
    if v < 1000:
        return f"{v:.3f} {unit}"
    v /= 1000
    for sign in ['K', 'M', 'G']:
        if v < 1000:
            return f"{v:.3f} {sign}{unit}"
        v /= 1000
    return f"{v:.3f} T{unit}"


class LinearGenerator(nn.Module):
    def __init__(self, resolution, n_classes, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3 * resolution * resolution),
            nn.Tanh(),
        )
        self.resolution = resolution

    def forward(self, z):
        x = self.main(z)
        x = x.view(-1, 3, self.resolution, self.resolution)
        return x


class LinearDiscriminator(nn.Module):
    def __init__(self, resolution, n_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3 * resolution * resolution, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        y = self.main(x)
        return y


def raw_D(D, x):
    return D(x)


def gn_D(D, x):
    return normalize_D(D, x, None)[0]


def estimate_time(G, D, shape_G, bs_G, forwar_D_fn, N):
    forw_G = 0
    forw_D = 0
    back_G = 0
    back_D = 0
    with trange(N, ncols=0, leave=False) as pbar:
        for n in pbar:
            z = torch.randn(bs_G, *shape_G, requires_grad=True).cuda()

            torch.cuda.synchronize()
            t1 = time.time()

            x = G(z)

            torch.cuda.synchronize()
            t2 = time.time()

            y = forwar_D_fn(D, x).mean()

            torch.cuda.synchronize()
            t3 = time.time()

            x_grad = torch.autograd.grad(y, x, torch.ones_like(y))[0]

            torch.cuda.synchronize()
            t4 = time.time()

            x.backward(x_grad)

            torch.cuda.synchronize()
            t5 = time.time()

            forw_G += t2 - t1
            forw_D += t3 - t2
            back_D += t4 - t3
            back_G += t5 - t4

            pbar.set_description(", ".join([
                f"forw_G: {forw_G * 1000 / (n + 1):.3f} ms",
                f"forw_D: {forw_D * 1000 / (n + 1):.3f} ms",
                f"back_G: {back_G * 1000 / (n + 1):.3f} ms",
                f"back_D: {back_D * 1000 / (n + 1):.3f} ms",
            ]))
    return forw_G, forw_D, back_G, back_D


def estimate_flops(G, D, shape_G, forwar_D_fn):
    z = torch.randn(1, *shape_G, requires_grad=True).cuda()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True
    ) as prof:
        x = G(z)
    events = prof.events()
    forw_G = sum([int(evt.flops) for evt in events])
    forw_G = si_format(forw_G, unit="FLOPs")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True
    ) as prof:
        y = forwar_D_fn(D, x)
    events = prof.events()
    forw_D = sum([int(evt.flops) for evt in events])
    forw_D = si_format(forw_D, unit="FLOPs")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True
    ) as prof:
        x_grad = torch.autograd.grad(y, x, torch.ones_like(y))[0]
    events = prof.events()
    back_D = sum([int(evt.flops) for evt in events])
    back_D = si_format(back_D, unit="FLOPs")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True
    ) as prof:
        x.backward(x_grad)
    events = prof.events()
    back_G = sum([int(evt.flops) for evt in events])
    back_G = si_format(back_G, unit="FLOPs")

    return forw_G, forw_D, back_G, back_D


if __name__ == '__main__':
    MODELS_ARGS = [
        {
            "G": (LinearGenerator, (32, None, 128), (128,), 128),
            "D": (LinearDiscriminator, (32, None), (3, 32, 32,), 128),
        },
        {
            "G": (dcgan.Generator, (32, None, 128), (128,), 128),
            "D": (dcgan.Discriminator, (32, None), (3, 32, 32,), 128),
        },
        {
            "G": (dcgan.Generator, (48, None, 128), (128,), 128),
            "D": (dcgan.Discriminator, (48, None), (3, 48, 48,), 128),
        },
        {
            "G": (resnet.Generator, (32, None, 128), (128,), 128),
            "D": (resnet.Discriminator, (32, None), (3, 32, 32,), 128),
        },
        {
            "G": (resnet.Generator, (48, None, 128), (128,), 128),
            "D": (resnet.Discriminator, (48, None), (3, 48, 48,), 128),
        },
        {
            "G": (resnet.Generator, (256, None, 128), (128,), 8),
            "D": (resnet.Discriminator, (256, None), (3, 256, 256,), 8),
        },
    ]

    N = 1000
    print('-' * 80)
    for args in MODELS_ARGS:
        MODEL_G, ARGS_G, shape_G, bs_G = args['G']
        G = MODEL_G(*ARGS_G).cuda()
        params_G = 0
        for param in G.parameters():
            params_G += param.numel()
        params_G = si_format(params_G)

        MODEL_D, ARGS_D, shape_D, bs_D = args['D']
        D = MODEL_D(*ARGS_D).cuda()
        params_D = 0
        for param in G.parameters():
            params_D += param.numel()
        params_D = si_format(params_D)

        print(f'name      : {MODEL_G.__module__}.{MODEL_G.__name__}{ARGS_G}')
        print(f'Params    : {params_G}')
        print(f'name      : {MODEL_D.__module__}.{MODEL_D.__name__}{ARGS_D}')
        print(f'Params    : {params_D}')

        G.requires_grad_(True)
        D.requires_grad_(False)

        forw_G, forw_D, back_G, back_D = estimate_flops(G, D, shape_G, raw_D)
        print(f'Raw Complexity:')
        print("\n".join([
            f"    forw_G    : {forw_G}",
            f"    forw_D    : {forw_D}",
            f"    back_G    : {back_G}",
            f"    back_D    : {back_D}",
        ]))
        forw_G, forw_D, back_G, back_D = estimate_flops(G, D, shape_G, gn_D)
        print(f'GN Complexity:')
        print("\n".join([
            f"    forw_G    : {forw_G}",
            f"    forw_D    : {forw_D}",
            f"    back_G    : {back_G}",
            f"    back_D    : {back_D}",
        ]))

        # D = dcgan.Discriminator(32, None).cuda()
        forw_G, forw_D, back_G, back_D = estimate_time(G, D, shape_G, bs_G, raw_D, N)
        print("Raw Speed:")
        print("\n".join([
            f"    forw_G    : {forw_G * 1000 / (N + 1):.3f} (ms/iter)",
            f"    forw_D    : {forw_D * 1000 / (N + 1):.3f} (ms/iter)",
            f"    back_G    : {back_G * 1000 / (N + 1):.3f} (ms/iter)",
            f"    back_D    : {back_D * 1000 / (N + 1):.3f} (ms/iter)",
        ]))

        forw_G, forw_D, back_G, back_D, = estimate_time(G, D, shape_G, bs_G, gn_D, N)
        print("GN Speed:")
        print("\n".join([
            f"    forw_G    : {forw_G * 1000 / (N + 1):.3f} (ms/iter)",
            f"    forw_D    : {forw_D * 1000 / (N + 1):.3f} (ms/iter)",
            f"    back_G    : {back_G * 1000 / (N + 1):.3f} (ms/iter)",
            f"    back_D    : {back_D * 1000 / (N + 1):.3f} (ms/iter)",
        ]))
        print('-' * 80)
