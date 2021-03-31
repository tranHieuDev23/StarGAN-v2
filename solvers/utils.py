from torch import Tensor
import torchvision.utils as vutils


def denormalize(x: Tensor):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol: int, filename: str):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
