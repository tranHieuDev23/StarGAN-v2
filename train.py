from solvers.plot import add_plot_point
import time
import datetime
from solvers.samples import sample_starganv2, video_starganv2
from solvers.load_data import InputFetcher, get_reference_loader, get_source_loader
from solvers.learner import StarGANv2Learner


def train_starganv2(gan_learner: StarGANv2Learner):
    args = gan_learner.args
    src_loader, classes = get_source_loader(
        args.dataset_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    src_fetcher = InputFetcher(src_loader)
    ref_fetcher = InputFetcher(get_reference_loader(
        args.dataset_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )[0])

    if (args.resume_iter > 0):
        print("Loading checkpoints at iteration {}...".format(args.resume_iter))
        gan_learner.load(args.resume_iter + 1)
        # args.resume_iter += 1

    print("Start training from iteration {}!".format(args.resume_iter))
    start_time = last_print_time = time.time()
    losses_list = dict()
    for iter in range(args.resume_iter, args.total_iters):
        print("Iteration {}".format(iter + 1))
        losses = gan_learner.train_step(src_fetcher, ref_fetcher)
        args.resume_iter = iter

        for item in losses:
            if (item not in losses_list):
                losses_list[item] = []
            losses_list[item].append(losses[item])

        if ((iter + 1) % args.print_every == 0):
            current_time = time.time()
            total_elapsed_time = str(datetime.timedelta(
                seconds=current_time - start_time))[:-7]
            time_from_last_print = str(datetime.timedelta(
                seconds=current_time - last_print_time))[:-7]
            print("Total elapsed time: {}; Time from last print: {}".format(
                total_elapsed_time, time_from_last_print
            ))
            print("step: {}; losses: {}".format(iter + 1, losses))
            last_print_time = current_time

        if ((iter + 1) % args.sample_every == 0):
            print("Sampling...")
            sample_starganv2(gan_learner, iter, src_fetcher, classes)
            for item in losses_list:
                value = sum(losses_list[item]) / len(losses_list[item])
                add_plot_point(args, item, iter + 1, value)
                losses_list[item] = []

        if ((iter + 1) % args.video_every == 0):
            print("Generating video...")
            video_starganv2(gan_learner, iter, src_fetcher)

        if ((iter + 1) % args.save_every == 0):
            print("Saving checkpoint...")
            gan_learner.save(iter + 1)
