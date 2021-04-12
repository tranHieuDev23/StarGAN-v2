from solvers.learner import StarGANv2Learner
from train import train_starganv2
from solvers.solver import StarGANv2
from solvers.stargan_args import StarGanArgs
import click


@click.command()
@click.option("--img_size", default=StarGanArgs.default_img_size, help="The two sizes of input images")
@click.option("--style_dim", default=StarGanArgs.default_style_dim, help="Size of style vector")
@click.option("--latent_dim", default=StarGanArgs.default_latent_dim, help="Size of latent vector")
@click.option("--num_domains", default=StarGanArgs.default_num_domains, help="Number of image domains")
@click.option("--w_hpf", default=StarGanArgs.default_w_hpf, help="Weight for high-pass filtering")
@click.option("--lambda_sty", default=StarGanArgs.default_lambda_sty, help="Weight for style reconstruction loss")
@click.option("--lambda_ds", default=StarGanArgs.default_lambda_ds, help="Weight for diversity sensitive loss")
@click.option("--lambda_cyc", default=StarGanArgs.default_lambda_cyc, help="Weight for cyclic consistency loss")
@click.option("--lambda_reg", default=StarGanArgs.default_lambda_reg, help="Weight for R1 regularization")
@click.option("--lr", default=StarGanArgs.default_lr, help="Learning rate for G, E and D models")
@click.option("--f_lr", default=StarGanArgs.default_f_lr, help="Learning rate for F model")
@click.option("--beta1", default=StarGanArgs.default_beta1, help="Decay rate for 1st moment of Adam")
@click.option("--beta2", default=StarGanArgs.default_beta2, help="Decay rate for 2nd moment of Adam")
@click.option("--weight_decay", default=StarGanArgs.default_weight_decay, help="Weight decay for optimizer")
@click.option("--dataset_dir", default=StarGanArgs.default_dataset_dir, help="Root directory of dataset")
@click.option("--resume_iter", default=StarGanArgs.default_resume_iter, help="Iteration to resume from. 0 to start from scratch.")
@click.option("--ds_iter", default=StarGanArgs.default_ds_iter, help="Number of iterations to optimize diversity sensitive loss")
@click.option("--total_iters", default=StarGanArgs.default_total_iters, help="Number of total iterations")
@click.option("--print_every", default=StarGanArgs.default_print_every, help="Frequency to log out losses")
@click.option("--sample_every", default=StarGanArgs.default_sample_every, help="Frequency to sample")
@click.option("--save_every", default=StarGanArgs.default_save_every, help="Frequency to save models")
@click.option("--eval_every", default=StarGanArgs.default_eval_every, help="Frequency to evaluate the models")
@click.option("--batch_size", default=StarGanArgs.default_batch_size, help="Batch size")
@click.option("--num_workers", default=StarGanArgs.default_num_workers, help="Number of worker threads to load the dataset")
@click.option("--checkpoint_dir", default=StarGanArgs.default_checkpoint_dir, help="Root directory to save the models")
@click.option("--sample_dir", default=StarGanArgs.default_sample_dir, help="Root directory to save sample images during training")
@click.option("--outputs_per_domain", default=StarGanArgs.default_outputs_per_domain, help="Number of images to generate via latent per domain during training sampling")
def main(img_size, style_dim, latent_dim, num_domains, w_hpf, lambda_sty, lambda_ds,
         lambda_cyc, lambda_reg, lr, f_lr, beta1, beta2, weight_decay, dataset_dir,
         resume_iter, ds_iter, total_iters, print_every, sample_every, save_every, eval_every,
         checkpoint_dir, batch_size, num_workers, sample_dir, outputs_per_domain):
    args = StarGanArgs()
    args.img_size = img_size
    args.style_dim = style_dim
    args.latent_dim = latent_dim
    args.num_domains = num_domains
    args.w_hpf = w_hpf
    args.lambda_sty = lambda_sty
    args.lambda_ds = lambda_ds
    args.lambda_cyc = lambda_cyc
    args.lambda_reg = lambda_reg
    args.lr = lr
    args.f_lr = f_lr
    args.beta1 = beta1
    args.beta2 = beta2
    args.weight_decay = weight_decay
    args.dataset_dir = dataset_dir
    args.resume_iter = resume_iter
    args.ds_iter = ds_iter
    args.total_iters = total_iters
    args.print_every = print_every
    args.sample_every = sample_every
    args.save_every = save_every
    args.eval_every = eval_every
    args.checkpoint_dir = checkpoint_dir
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.sample_dir = sample_dir
    args.outputs_per_domain = outputs_per_domain

    starGan = StarGANv2Learner(args)
    train_starganv2(starGan)


if __name__ == "__main__":
    main()
