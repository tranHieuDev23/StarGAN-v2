from PIL import ImageCms
import numpy as np

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
    srgb_profile, lab_profile, "RGB", "LAB")
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(
    lab_profile, srgb_profile, "LAB", "RGB")


def rgb_to_lab(image):
    return ImageCms.applyTransform(image, rgb2lab_transform)


def lab_to_rgb(image):
    return ImageCms.applyTransform(image, lab2rgb_transform)
