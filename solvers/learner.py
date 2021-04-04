import os
import torch
from munch import Munch
from solvers.stargan_args import StarGanArgs
from solvers.solver import StarGANv2
from solvers.losses import compute_discriminator_loss, compute_generator_loss
from solvers.load_data import InputFetcher
from solvers.checkpoint import CheckpointHandler


class StarGANv2Learner(StarGANv2):
    def __init__(self, args: StarGanArgs):
        super().__init__(args)
        self.generator_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=args.lr, betas=[args.beta1, args.beta2],
            weight_decay=args.weight_decay
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            params=self.mapping_network.parameters(),
            lr=args.f_lr, betas=[args.beta1, args.beta2],
            weight_decay=args.weight_decay
        )
        self.style_encoder_optimizer = torch.optim.Adam(
            params=self.style_encoder.parameters(),
            lr=args.lr, betas=[args.beta1, args.beta2],
            weight_decay=args.weight_decay
        )
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=args.lr, betas=[args.beta1, args.beta2],
            weight_decay=args.weight_decay
        )
        self.optimizer_checkpoint_handler = CheckpointHandler(
            os.path.join(args.checkpoint_dir, "optimizers_{:06d}.ckpt"),
            Munch(
                generator=self.generator_optimizer,
                mapping_network=self.mapping_network_optimizer,
                style_encoder=self.style_encoder_optimizer,
                discriminator=self.discriminator_optimizer
            )
        )

    def save(self, step):
        super().save(step)
        self.optimizer_checkpoint_handler.save(step)

    def load(self, step):
        super().load(step)
        self.optimizer_checkpoint_handler.load(step)

    def _reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.mapping_network.zero_grad()
        self.style_encoder.zero_grad()
        self.discriminator.zero_grad()

    def train_step(self, source_fetcher: InputFetcher, ref_fetcher: InputFetcher):
        args = self.args

        x_real, y_org = next(source_fetcher)
        x_real, y_org = x_real.to(self.device), y_org.to(self.device)
        x_ref_1, x_ref_2, y_ref = next(ref_fetcher)
        x_ref_1, x_ref_2, y_ref = x_ref_1.to(self.device), x_ref_2.to(
            self.device), y_ref.to(self.device)
        z_1 = torch.randn(x_real.size(0), args.latent_dim).to(self.device)
        z_2 = torch.randn(x_real.size(0), args.latent_dim).to(self.device)

        d_loss, d_loss_latent = compute_discriminator_loss(
            args, x_real, y_org, y_ref, self.generator, self.style_encoder,
            self.mapping_network, self.discriminator, z_target=z_1)
        self._reset_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        d_loss, d_loss_refer = compute_discriminator_loss(
            args, x_real, y_org, y_ref, self.generator, self.style_encoder,
            self.mapping_network, self.discriminator, x_refer=x_ref_1)
        self._reset_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        g_loss, g_loss_latent = compute_generator_loss(
            args, x_real, y_org, y_ref, self.generator, self.style_encoder,
            self.mapping_network, self.discriminator, z_targets=[z_1, z_2])
        self._reset_grad()
        g_loss.backward()
        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()
        self.style_encoder_optimizer.step()

        g_loss, g_loss_refer = compute_generator_loss(
            args, x_real, y_org, y_ref, self.generator, self.style_encoder,
            self.mapping_network, self.discriminator, x_refers=[x_ref_1, x_ref_2])
        self._reset_grad()
        g_loss.backward()
        self.generator_optimizer.step()

        actual_lambda_ds = max(args.lambda_ds * (1 - args.resume_iter / args.ds_iter), 0)

        all_losses = dict()
        for loss, prefix in zip([d_loss_latent, d_loss_refer, g_loss_latent, g_loss_refer],
                                ["D/latent_", "D/ref_", "G/latent_", "G/ref_"]):
            for key, value in loss.items():
                all_losses[prefix + key] = value
        all_losses["G/lambda_ds"] = actual_lambda_ds
        return all_losses
