import torch
import torch.nn as nn
import numpy as np
from models.blocks import ResidualBlock


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        layers = []
        layers.append(nn.Conv2d(3, dim_in, 1, 1, 0))

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            layers.append(ResidualBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(dim_out, dim_in, 4, 1, 0))
        layers.append(nn.LeakyReLU(0.2))
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        output = []
        for layer in self.unshared:
            output.append(layer(h))
        output = torch.stack(output, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = output[idx, y]
        return s
