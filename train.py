import time
import datetime
from solvers.samples import sample_starganv2
from solvers.load_data import InputFetcher, get_reference_loader, get_source_loader
from solvers.learner import StarGANv2Learner


def train_starganv2(gan_learner: StarGANv2Learner):
    args = gan_learner.args
    src_fetcher = InputFetcher(get_source_loader(
        args.dataset_dir, img_size=args.img_size
    )[0])
    ref_fetcher = InputFetcher(get_reference_loader(
        args.dataset_dir, img_size=args.img_size
    )[0])

    if (args.resume_iter > 0):
        print("Loading checkpoints at iteration {}...".format(args.resume_iter))
        gan_learner.load(args.resume_iter)

    print("Start training from iteration {}!".format(args.resume_iter))
    start_time = last_print_time = time.time()
    for iter in range(args.resume_iter, args.total_iters):
        print("Iteration {}".format(iter + 1))
        args.resume_iter = iter
        losses = gan_learner.train_step(src_fetcher, ref_fetcher)

        if ((iter + 1) % args.save_every == 0):
            print("Saving checkpoint...")
            gan_learner.save(iter + 1)

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
            sample_starganv2(gan_learner)
