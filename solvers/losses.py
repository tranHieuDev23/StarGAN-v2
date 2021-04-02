from models.discriminator import Discriminator
from models.maping_network import MappingNetwork
from typing import List
from munch import Munch
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from solvers.stargan_args import StarGanArgs
from models.style_encoder import StyleEncoder
from models.generator import Generator

REAL_LABEL = 1
FAKE_LABEL = 0


def compute_adversarial_loss(logits: Tensor, target: int):
    if (target not in [REAL_LABEL, FAKE_LABEL]):
        raise RuntimeError("Invalid target value: {}".format(target))
    targets = torch.full_like(logits, fill_value=target)
    return binary_cross_entropy_with_logits(logits, targets)


def compute_style_reconstruction_loss(x_fake: Tensor, y_target: Tensor,
                                      s_target: Tensor, style_encoder: StyleEncoder):
    s_pred = style_encoder(x_fake, y_target)
    return torch.mean(torch.abs(s_pred - s_target))


def compute_diversity_sensitive_loss(x_fake_1: Tensor, x_fake_2: Tensor):
    return torch.mean(torch.abs(x_fake_1 - x_fake_2))


def compute_cycle_consistency_loss(x_real: Tensor, y_org: Tensor,
                                   x_fake: Tensor, generator: Generator,
                                   style_encoder: StyleEncoder):
    s_org = style_encoder(x_real, y_org)
    x_recreation = generator(x_fake, s_org)
    return torch.mean(torch.abs(x_recreation - x_real))


def compute_r1_regulation(logits: Tensor, x_real: Tensor):
    # Zero-centered gradient penalty for real images
    batch_size = x_real.size(0)
    grad_logits = torch.autograd.grad(
        outputs=logits.sum(), inputs=x_real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_logits2 = grad_logits.pow(2)
    return 0.5 * grad_logits2.view(batch_size, -1).sum(1).mean(0)


def compute_discriminator_loss(args: StarGanArgs, x_real: Tensor, y_org: Tensor,
                               y_target: Tensor, generator: Generator, style_encoder: StyleEncoder,
                               mapping_network: MappingNetwork, discriminator: Discriminator,
                               z_target: Tensor = None, x_refer: Tensor = None):
    if (z_target is None) == (x_refer is None):
        raise RuntimeError("Either z_target or x_refer must be provided!")
    # With real images
    x_real.requires_grad_()
    output_real = discriminator(x_real, y_org)
    loss_real = compute_adversarial_loss(output_real, REAL_LABEL)
    loss_regulation = compute_r1_regulation(output_real, x_real)

    # With fake images
    with torch.no_grad():
        if (z_target is not None):
            s_target = mapping_network(z_target, y_target)
        else:
            s_target = style_encoder(x_refer, y_target)
        x_fake = generator(x_real, s_target)
    output_fake = discriminator(x_fake, y_target)
    loss_fake = compute_adversarial_loss(output_fake, FAKE_LABEL)

    loss = loss_real + loss_fake + args.lambda_reg * loss_regulation
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       regulation=loss_regulation.item())


def compute_generator_loss(args: StarGanArgs, x_real: Tensor, y_org: Tensor,
                           y_target: Tensor, generator: Generator, style_encoder: StyleEncoder,
                           mapping_network: MappingNetwork, discriminator: Discriminator,
                           z_targets: List[Tensor] = None, x_refers: List[Tensor] = None):
    if (z_targets is None) == (x_refers is None):
        raise RuntimeError("Either z_targets or x_refers must be provided!")

    if (z_targets is not None):
        s_targets = [mapping_network(item, y_target) for item in z_targets]
    else:
        s_targets = [style_encoder(item, y_target) for item in x_refers]
    x_fakes = [generator(x_real, item) for item in s_targets]

    output_fake_1 = discriminator(x_fakes[0], y_target)
    loss_adversarial = compute_adversarial_loss(output_fake_1, REAL_LABEL)

    loss_style_reconstruction = compute_style_reconstruction_loss(
        x_fakes[0], y_target, s_targets[0], style_encoder)

    loss_diversity_sensitive = compute_diversity_sensitive_loss(
        x_fakes[0], x_fakes[1])

    loss_cycle_consistency = compute_cycle_consistency_loss(
        x_real, y_org, x_fakes[0], generator, style_encoder)

    actual_lambda_ds = max(
        args.lambda_ds * (1 - args.resume_iter / args.ds_iter), 0)

    loss = loss_adversarial + args.lambda_sty * loss_style_reconstruction \
        - actual_lambda_ds * loss_diversity_sensitive \
        + args.lambda_cyc * loss_cycle_consistency

    return loss, Munch(adversarial=loss_adversarial.item(),
                       style_reconstruction=loss_style_reconstruction.item(),
                       diversity_sensitive=loss_diversity_sensitive.item(),
                       cycle_consistency=loss_cycle_consistency.item())
