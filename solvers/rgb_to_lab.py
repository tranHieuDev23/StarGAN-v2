from PIL.Image import Image
from torch import tensor
import torch
from torchvision import transforms
from skimage import color
import numpy as np


def rgb_to_lab(image: Image):
    image = np.array(image)
    image = color.rgb2lab(image)
    return image


def lab_normalize():
    return transforms.Normalize(mean=[50, 0, 0], std=[50, 128, 128])


def lab_denormalize():
    return transforms.Normalize(mean=[-1, 0, 0], std=[1 / 50, 1 / 128, 1 / 128])


def lab_to_rgb(image: np.array):
    image = color.lab2rgb(image)
    return image


def _tensor_to_rgb_(x: tensor):
    N, C, H, W = x.shape
    xs = [lab_denormalize()(x[i]) for i in range(N)]
    xs = [item.permute(1, 2, 0).cpu().numpy() for item in xs]
    xs = [lab_to_rgb(item) for item in xs]
    xs = [transforms.ToTensor()(item) for item in xs]
    x = torch.stack(xs, dim=0)
    x = x.permute(0, 1, 2, 3)
    return x
