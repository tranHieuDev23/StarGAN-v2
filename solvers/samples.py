import os
from os import path
import shutil
from typing import List
from solvers.utils import save_image
import torch
from torch import Tensor
from solvers.solver import StarGANv2


def _sample_cycle_consistency_(gan: StarGANv2, x_real: Tensor, y_real: Tensor,
                               x_refer: Tensor, y_refer: Tensor, filename: str):
    s_refer = gan.generate_image_style(x_refer, y_refer)
    x_fake = gan.generate_image_with_style(x_real, s_refer)
    s_real = gan.generate_image_style(x_real, y_real)
    x_recreation = gan.generate_image_with_style(x_fake, s_real)
    x_concat = torch.cat([x_real, x_refer, x_fake, x_recreation], dim=0)
    save_image(x_concat, filename)


def _sample_latent_synthensis_(gan: StarGANv2, x_real: Tensor, y_target: Tensor, z_target_list: List[Tensor], filename: str, psi: float):
    x_concat = [x_real]
    for z_target in z_target_list:
        x_fake = gan.generate_image_from_latent(
            x_real, y_target, z_target, psi)
        x_concat.append(x_fake)
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, filename)


def sample_starganv2(gan: StarGANv2, step: int, x_real: Tensor,
                     y_real: Tensor, x_refer: Tensor, y_refer: Tensor):
    args = gan.args
    N = x_real.size(0)
    sample_dir = path.join(args.sample_dir, str(step))
    shutil.rmtree(sample_dir, ignore_errors=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Sample cycle-consistency
    filename = path.join(sample_dir, "cycle_consistency.jpg")
    _sample_cycle_consistency_(gan, x_real, y_real, x_refer, y_refer, filename)

    # Image synthensis based on latent vector
    y_target_list = [torch.full(N, y).to(gan.device)
                     for y in range(args.num_domains)]
    z_target_list = [torch.randn(1, args.latent_dim).repeat(N, 1).to(gan.device)
                     for _ in range(args.outputs_per_domain)]
    for psi in [0.5, 0.7, 1.0]:
        for y_idx, y_target in y_target_list:
            filename = path.join(
                sample_dir, "latent_synthensis_target_  {}_psi_{}.jpg".format(y_idx, psi))
            _sample_latent_synthensis_(
                gan, x_real, y_target, z_target_list, filename, psi)
