import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, d=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d * 4, d * 8, 4, 1, 1),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d * 8, num_domains, 4, 1, 1)
        )

    def forward(self, x, y):
        output = self.main(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        output = output[idx, y]
        output = output.view(output.size(0), -1)
        output = torch.mean(output, dim=1)[0]
        return output
