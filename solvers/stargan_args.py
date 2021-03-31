class StarGanArgs:
    default_img_size = 512
    default_style_dim = 64
    default_latent_dim = 16
    default_num_domains = 4

    default_w_hpf = 0
    default_lambda_sty = 1
    default_lambda_ds = 1
    default_lambda_cyc = 1
    default_lambda_reg = 1

    default_lr = 1e-4
    default_f_lr = 1e-6
    default_beta1 = 0
    default_beta2 = 0.99
    default_weight_decay = 1e6

    default_dataset_dir = "dataset"
    default_resume_iter = 0
    default_ds_iter = 100000
    default_total_iters = 100000
    default_print_every = 100
    default_sample_every = 100
    default_save_every = 100
    default_eval_every = 1000
    default_batch_size = 8
    default_num_workers = 4
    default_outputs_per_domain = 5
    default_checkpoint_dir = "checkpoints"
    default_sample_dir = "samples"

    def __init__(self):
        self.img_size = StarGanArgs.default_img_size
        self.style_dim = StarGanArgs.default_style_dim
        self.latent_dim = StarGanArgs.default_latent_dim
        self.num_domains = StarGanArgs.default_num_domains

        self.w_hpf = StarGanArgs.default_w_hpf
        self.lambda_sty = StarGanArgs.default_lambda_sty
        self.lambda_ds = StarGanArgs.default_lambda_ds
        self.lambda_cyc = StarGanArgs.default_lambda_cyc
        self.lambda_reg = StarGanArgs.default_lambda_reg

        self.lr = StarGanArgs.default_lr
        self.f_lr = StarGanArgs.default_f_lr
        self.beta1 = StarGanArgs.default_beta1
        self.beta2 = StarGanArgs.default_beta2
        self.weight_decay = StarGanArgs.default_weight_decay

        self.dataset_dir = StarGanArgs.default_dataset_dir
        self.resume_iter = StarGanArgs.default_resume_iter
        self.ds_iter = StarGanArgs.default_ds_iter
        self.total_iters = StarGanArgs.default_total_iters
        self.print_every = StarGanArgs.default_print_every
        self.sample_every = StarGanArgs.default_sample_every
        self.save_every = StarGanArgs.default_save_every
        self.eval_every = StarGanArgs.default_eval_every
        self.batch_size = StarGanArgs.default_batch_size
        self.num_workers = StarGanArgs.default_num_workers
        self.checkpoint_dir = StarGanArgs.default_checkpoint_dir
        self.sample_dir = StarGanArgs.default_sample_dir
        self.outputs_per_domain = StarGanArgs.default_outputs_per_domain
