import os
from os import path
import shutil
from solvers.rgb_to_ycbcr import _tensor_to_rgb_

from solvers.video import video_latent, video_ref
from solvers.load_data import InputFetcher
from typing import List
from solvers.utils import save_image
import torch
from torch import Tensor
from solvers.solver import StarGANv2


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


def _sample_latent_synthensis_(gan: StarGANv2, x_real: Tensor, y_real, s_target: Tensor, filename: str):
    N, C, H, W = x_real.size()
    x_concat = [x_real]
    s_real = gan.generate_image_style(x_real, y_real)
    s_target = s_target.repeat(N, 1)
    for psi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s_fused = torch.lerp(s_real, s_target, psi)
        x_fake = gan.generate_image_with_style(x_real, s_fused)
        x_concat.append(x_fake)
    x_concat = [_tensor_to_rgb_(item) for item in x_concat]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


def sample_starganv2(gan: StarGANv2, step: int, source_fetcher: InputFetcher, classes: List[str]):
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
    s_target_list = [gan.generate_domain_average_style(
        item) for item in y_target_list]
    for y_idx, s_target in enumerate(s_target_list):
        filename = path.join(
            sample_dir, "latent_synthensis_target_{}.jpg".format(classes[y_idx]))
        _sample_latent_synthensis_(
            gan, x_real, y_real, s_target, filename)


def video_starganv2(gan: StarGANv2, step: int, source_fetcher: InputFetcher):
    args = gan.args
    x_real, y_real = next(source_fetcher)
    x_real, y_real = x_real.to(gan.device), y_real.to(gan.device)
    x_refer, y_refer = next(source_fetcher)
    x_refer, y_refer = x_refer.to(gan.device), y_refer.to(gan.device)

    sample_dir = path.join(args.sample_dir, str(step))
    os.makedirs(sample_dir, exist_ok=True)

    # Image synthensis based on latent vector
    y_target_list = [torch.tensor(y).repeat(1).to(
        gan.device) for y in range(args.num_domains)]
    z_target_list = [gan.generate_domain_average_style(
        item) for item in y_target_list]
    # Video generation with reference
    filename = path.join(sample_dir, f"video_ref.mp4")
    video_ref(gan, x_real, x_refer, y_refer, filename)
    # Video generation with latent vector
    filename = path.join(sample_dir, f"video_latent.mp4")
    video_latent(gan, x_real, z_target_list, filename)
