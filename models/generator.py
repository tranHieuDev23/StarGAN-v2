import torch
import torch.nn as nn
import numpy as np
from models.blocks import AdaInResBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_lab = nn.Conv2d(3, dim_in, 1, 1, 0)
        self._build_encoder_decoder(dim_in, style_dim, max_conv_dim)
        self.to_ab = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 2, 1, 1, 0))

    def _build_encoder_decoder(self, dim_in, style_dim, max_conv_dim):
        self.encode = nn.ModuleList()
        self.decode_ab = nn.ModuleList()

        repeat_num = int(np.log2(self.img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResidualBlock(dim_in, dim_out, normalize=True, downsample=True))
            self.decode_ab.insert(
                0, AdaInResBlock(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(
                ResidualBlock(dim_out, dim_out, normalize=True))
            self.decode_ab.insert(
                0, AdaInResBlock(dim_out, dim_out, style_dim))

    def forward(self, lab, s):
        l = lab[:, 0:1]
        x = self.from_lab(lab)
        for block in self.encode:
            x = block(x)
        for block in self.decode_ab:
            x = block(x, s)
        ab = self.to_ab(x)
        return torch.cat([l, ab], dim=1)
