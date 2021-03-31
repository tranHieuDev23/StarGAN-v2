import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, 512))
        layers.append(nn.ReLU())
        for _ in range(3):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, style_dim)
            ))

    def forward(self, z, y):
        h = self.shared(z)
        output = []
        for layer in self.unshared:
            output.append(layer(h))
        output = torch.stack(output, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = output[idx, y]
        return s
