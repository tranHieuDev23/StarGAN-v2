import os
from os import path
import shutil
from PIL import Image
import numpy as np

from solvers.rgb_to_lab import lab_to_rgb
from solvers.load_data import InputFetcher
from typing import List
from solvers.utils import save_image
import torch
from torchvision import transforms
from torch import Tensor
from solvers.solver import StarGANv2


def _tensor_to_rgb_(x: Tensor):
    N, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)
    xs = [x[i].cpu().numpy() for i in range(N)]
    xs = [Image.fromarray(item.astype("uint8"), mode="LAB") for item in xs]
    xs = [lab_to_rgb(item) for item in xs]
    xs = [transforms.ToTensor()(item) for item in xs]
    x = torch.stack(xs, dim=0)
    x = x.permute(0, 1, 2, 3)
    return x


def _sample_cycle_consistency_(gan: StarGANv2, x_real: Tensor, y_real: Tensor,
                               x_refer: Tensor, y_refer: Tensor, filename: str):
    N, C, H, W = x_real.size()
    s_refer = gan.generate_image_style(x_refer, y_refer)
    x_fake = gan.generate_image_with_style(x_real, s_refer)
    s_real = gan.generate_image_style(x_real, y_real)
    x_recreation = gan.generate_image_with_style(x_fake, s_real)

    x_concat = [x_real, x_refer, x_fake, x_recreation]
    x_concat = [_tensor_to_rgb_(item) for item in x_concat]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


def _sample_latent_synthensis_(gan: StarGANv2, x_real: Tensor, y_target: Tensor, z_target_list: List[Tensor], filename: str, psi: float):
    N, C, H, W = x_real.size()
    x_concat = [x_real]
    for z_target in z_target_list:
        x_fake = gan.generate_image_from_latent(
            x_real, y_target, z_target, psi)
        x_concat.append(x_fake)

    x_concat = [_tensor_to_rgb_(item) for item in x_concat]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


def sample_starganv2(gan: StarGANv2, step: int, source_fetcher: InputFetcher):
    args = gan.args
    x_real, y_real = next(source_fetcher)

    x_real, y_real = x_real.to(gan.device), y_real.to(gan.device)
    x_refer, y_refer = next(source_fetcher)
    x_refer, y_refer = x_refer.to(gan.device), y_refer.to(gan.device)
    N = x_real.size(0)
    sample_dir = path.join(args.sample_dir, str(step))
    shutil.rmtree(sample_dir, ignore_errors=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Sample cycle-consistency
    filename = path.join(sample_dir, "cycle_consistency.jpg")
    _sample_cycle_consistency_(gan, x_real, y_real, x_refer, y_refer, filename)

    # Image synthensis based on latent vector
    y_target_list = [torch.tensor(y).repeat(1).to(
        gan.device) for y in range(args.num_domains)]
    z_target_list = [torch.randn(1, args.latent_dim).repeat(N, 1).to(gan.device)
                     for _ in range(args.outputs_per_domain)]
    for psi in [0.5, 0.7, 1.0]:
        for y_idx, y_target in enumerate(y_target_list):
            filename = path.join(
                sample_dir, "latent_synthensis_target_  {}_psi_{}.jpg".format(y_idx, psi))
            _sample_latent_synthensis_(
                gan, x_real, y_target, z_target_list, filename, psi)
