import torch
import torch.nn as nn
from torchviz import make_dot

from training.gn import normalize_D


if __name__ == '__main__':
    D = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.ReLU(),
    ).cuda()
    D.requires_grad_(False)

    x = torch.randn(1, 128, requires_grad=True).cuda()
    y = D(x).mean()
    make_dot(y, params=dict(D.named_parameters()), show_attrs=True, show_saved=True, max_attr_chars=99999).render("./vis/speed/D_graph", format="svg")

    # Case 2
    x = torch.randn(1, 128, requires_grad=True).cuda()
    y, _ = normalize_D(D, x, None)
    y = y.mean()
    make_dot(y, params=dict(D.named_parameters()), show_attrs=True, show_saved=True, max_attr_chars=99999).render("./vis/speed/D_GN_graph", format="svg")

    # Case 3
    x = torch.randn(1, 128, requires_grad=True).cuda()
    y = D(x)
    grad = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    grad_norm = grad.flatten(start_dim=1).norm(dim=1).view(-1, 1)
    y = grad_norm.mean()
    make_dot(y, params=dict(D.named_parameters()), show_attrs=True, show_saved=True, max_attr_chars=99999).render("./vis/speed/GP_graph", format="svg")
