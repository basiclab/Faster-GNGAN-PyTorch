import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = x.view(*self.shape)
        return x


class Rescalable(nn.Module):
    def rescale(self, base_scale=1., alpha=1.):
        raise NotImplementedError()


class RescalableSequentialModel(Rescalable):
    """Sequentially expand chidren modules."""
    def expand(self, p, base_scale, alpha):
        for m in p.children():
            if isinstance(m, Rescalable):
                base_scale = m.rescale(base_scale, alpha)
            else:
                base_scale = self.expand(m, base_scale, alpha)
        return base_scale

    def rescale(self, base_scale=1., alpha=1.):
        base_scale = self.expand(self, base_scale, alpha)
        return base_scale


class RescalableResBlock(Rescalable):
    def __init__(self):
        super().__init__()
        self.shortcut_scale = 1

    @torch.no_grad()
    def rescale(self, base_scale, alpha=1.):
        assert hasattr(self, 'shortcut'), ".shortcut is not defined in the derived class of RescalableResBlock"
        assert hasattr(self, 'main'), ".main is not defined in the derived class of RescalableResBlock"
        residual_scale = base_scale
        for module in self.main.modules():
            if isinstance(module, RescalableWrapper):
                residual_scale = module.rescale(residual_scale, alpha)

        shortcut_scale = base_scale
        for module in self.shortcut.modules():
            if isinstance(module, RescalableWrapper):
                shortcut_scale = module.rescale(shortcut_scale, alpha)
        self.shortcut_scale = residual_scale / shortcut_scale

        return residual_scale

    def forward(self, x):
        return self.main(x) + self.shortcut(x) * self.shortcut_scale


class RescalableWrapper(Rescalable):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        assert 'weight' in module._parameters

    def init_module(self):
        if 'weight' in self.module._parameters:
            self._init_param('weight')
        if 'bias' in self.module._parameters:
            self._init_param('bias')

    def _init_param(self, name):
        params = self.module._parameters[name]
        self.module.register_parameter(f"{name}_raw", params)
        self.module.register_buffer(f'{name}_scale', torch.ones(()))
        self.module.register_buffer(f'{name}_norm', params.data.norm(p=2))
        delattr(self.module, name)
        setattr(self.module, name, params.data)

    @torch.no_grad()
    def rescale(self, base_scale=1., alpha=1.):
        if 'weight_raw' in self.module._parameters:
            self.module.weight_scale = alpha * self.module.weight_norm / (
                self.module.weight_raw.norm(p=2) + 1e-12)
            base_scale = base_scale * self.module.weight_scale
        if 'bias_raw' in self.module._parameters:
            self.module.bias_scale = base_scale
        return base_scale

    def forward(self, *args, **kwargs):
        for name in ['weight', 'bias']:
            if f"{name}_raw" in self.module._parameters:
                param = self.module._parameters[f"{name}_raw"]
                scale = self.module._buffers[f'{name}_scale']
                setattr(self.module, name, param * scale)
        return self.module(*args, **kwargs)
