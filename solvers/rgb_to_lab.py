from PIL import Image, ImageCms
from torch import tensor
import torch
from torchvision import transforms

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
    srgb_profile, lab_profile, "RGB", "LAB")
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(
    lab_profile, srgb_profile, "LAB", "RGB")


def rgb_to_lab(image):
    return ImageCms.applyTransform(image, rgb2lab_transform)


def lab_normalize():
    return transforms.Normalize(mean=[50, 0, 0], std=[50, 128, 128])


def lab_denormalize():
    return transforms.Normalize(mean=[-1, 0, 0], std=[1 / 50, 1 / 128, 1 / 128])


def lab_to_rgb(image):
    return ImageCms.applyTransform(image, lab2rgb_transform)


def _tensor_to_rgb_(x: tensor):
    N, C, H, W = x.shape
    xs = [lab_denormalize()(x[i]) for i in range(N)]
    xs = [item.permute(1, 2, 0).cpu().numpy() for item in xs]
    xs = [Image.fromarray(item.astype("uint8"), mode="LAB") for item in xs]
    xs = [lab_to_rgb(item) for item in xs]
    xs = [transforms.ToTensor()(item) for item in xs]
    x = torch.stack(xs, dim=0)
    x = x.permute(0, 1, 2, 3)
    return x
