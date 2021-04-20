from PIL import Image
from torch import Tensor
import torch
from torchvision import transforms


def rgb_to_ycbcr(image: Image):
    return image.convert('YCbCr')


def ycbcr_denormalize():
    return transforms.Normalize(mean=[-1, 0, 0], std=[1 / 50, 1 / 128, 1 / 128])


def ycbcr_to_rgb(image):
    return image.convert('RGB')


def _tensor_to_rgb_(x: Tensor):
    N, C, H, W = x.shape
    xs = [transforms.ToPILImage(mode="YCbCr")(x[i]) for i in range(N)]
    xs = [item.convert("RGB") for item in xs]
    xs = [transforms.ToTensor()(item) for item in xs]
    x = torch.stack(xs, dim=0)
    return x
