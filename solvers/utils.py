from torch import Tensor
import torch.nn as nn
import torchvision.utils as vutils


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x: Tensor):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x: Tensor, ncol: int, filename: str):
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
