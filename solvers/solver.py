import os
from solvers.utils import he_init
import torch
import torch.nn as nn
from torch import Tensor
from munch import Munch
from solvers.stargan_args import StarGanArgs
from solvers.checkpoint import CheckpointHandler
from models.discriminator import Discriminator
from models.style_encoder import StyleEncoder
from models.maping_network import MappingNetwork
from models.generator import Generator


class StarGANv2(nn.Module):
    def __init__(self, args: StarGanArgs):
        super().__init__()
        self.args = args
        self.generator = Generator(
            args.img_size, args.style_dim, w_hpf=args.w_hpf)
        self.mapping_network = MappingNetwork(
            args.latent_dim, args.style_dim, args.num_domains)
        self.style_encoder = StyleEncoder(
            args.img_size, args.style_dim, args.num_domains)
        self.discriminator = Discriminator(
            args.img_size, args.num_domains)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.generator.apply(he_init)
        self.mapping_network.apply(he_init)
        self.style_encoder.apply(he_init)
        self.discriminator.apply(he_init)
        self.checkpoint_handler = CheckpointHandler(
            os.path.join(args.checkpoint_dir, "nets_{:06d}.ckpt"),
            Munch(
                generator=self.generator,
                mapping_network=self.mapping_network,
                style_encoder=self.style_encoder,
                discriminator=self.discriminator
            ))

    def save(self, step: int):
        self.checkpoint_handler.save(step)

    def load(self, step: int):
        self.checkpoint_handler.load(step)

    @torch.no_grad()
    def generate_image_style(self, x_src: Tensor, y_src: Tensor):
        return self.style_encoder(x_src, y_src)

    @torch.no_grad()
    def generate_latent_style(self, z_src: Tensor, y_src: Tensor):
        return self.mapping_network(z_src, y_src)

    @torch.no_grad()
    def generate_image_with_style(self, x_src: Tensor, z_target: Tensor):
        return self.generator(x_src, z_target)

    @torch.no_grad()
    def generate_domain_average_style(self, y_target: Tensor) -> Tensor:
        z_many = torch.randn(10000, self.args.latent_dim).to(self.device)
        y_many = torch.LongTensor(10000).to(self.device).fill_(y_target[0])
        s_many = self.generate_latent_style(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        return s_avg

    @torch.no_grad()
    def generate_image_from_latent(self, x_src: Tensor, y_target: Tensor, z_target: Tensor, psi: float = 1) -> Tensor:
        N, C, H, W = x_src.size()
        s_target = self.mapping_network(z_target, y_target)
        if (psi < 1):
            s_avg = self.generate_domain_average_style(y_target)
            s_target = torch.lerp(s_avg, s_target, psi)
        return self.generate_image_with_style(x_src, s_target)

    @torch.no_grad()
    def generate_image_from_reference(self, x_src: Tensor, y_target: Tensor, x_refer: Tensor) -> Tensor:
        s_target = self.style_encoder(x_refer, y_target)
        return self.generate_image_with_style(x_src, s_target)
