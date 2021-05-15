import math
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, conv2d

UNIT_VARIANCE = math.sqrt(2)


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()
        self.activation = activation
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_main_path(dim_in, dim_out, activation)
        self._build_shortcut(dim_in, dim_out)

    def _build_main_path(self, dim_in, dim_out, activation):
        modules = []
        if (self.normalize):
            modules.append(nn.InstanceNorm2d(dim_in, affine=True))
        modules.append(activation)
        modules.append(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        if (self.downsample):
            modules.append(nn.AvgPool2d(kernel_size=2))
        if (self.normalize):
            modules.append(nn.InstanceNorm2d(dim_in, affine=True))
        modules.append(activation)
        modules.append(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        self._main_path = nn.Sequential(*modules)

    def _build_shortcut(self, dim_in, dim_out):
        modules = []
        if (self.learned_sc):
            modules.append(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))
        if (self.downsample):
            modules.append(nn.AvgPool2d(kernel_size=2))
        self._shortcut = nn.Sequential(*modules)

    def forward(self, x):
        x = self._shortcut(x) + self._main_path(x)
        return x / UNIT_VARIANCE


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        x = self.norm(x)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * x + beta


class AdaInResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, activation=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.activation(x)
        if self.upsample:
            x = interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / UNIT_VARIANCE
        return out
