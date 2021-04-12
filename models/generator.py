import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from models.blocks import AdaInResBlock, HighPass, ResidualBlock


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 1, 1, 0)
        self._build_encoder_decoder(dim_in, style_dim, max_conv_dim, w_hpf)
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0)
        )
        if (w_hpf > 0):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.hpf = HighPass(w_hpf, device)

    def _build_encoder_decoder(self, dim_in, style_dim, max_conv_dim, w_hpf):
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        repeat_num = int(np.log2(self.img_size)) - 4
        if (w_hpf > 0):
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResidualBlock(dim_in, dim_out, normalize=True, downsample=True)
            )
            self.decode.insert(
                0, AdaInResBlock(dim_out, dim_in, style_dim,
                                 w_hpf, upsample=True)
            )
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(
                ResidualBlock(dim_out, dim_out, normalize=True)
            )
            self.decode.insert(
                0, AdaInResBlock(dim_out, dim_out, style_dim, w_hpf)
            )

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            feature_size = x.size(2)
            if (masks is not None and feature_size in [32, 64, 128, 256]):
                cache[feature_size] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            feature_size = x.size(2)
            if (masks is not None and feature_size in [32, 64, 128, 256]):
                mask = masks[0] if feature_size in [32] else masks[1]
                mask = interpolate(mask, size=feature_size, mode="bilinear")
                x = x + self.hpf(mask * cache[feature_size])
        return self.to_rgb(x)
