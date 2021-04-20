import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from solvers.stargan_args import StarGanArgs


def add_plot_point(args: StarGanArgs, plot_name: str, x: int, y: float):
    os.makedirs(args.plot_dir, exist_ok=True)
    csv_path = os.path.join(args.plot_dir, f"{plot_name}.csv")
    figure_path = os.path.join(args.plot_dir, f"{plot_name}.png")
    if (not os.path.isfile(csv_path)):
        with open(csv_path, "w"):
            pass
    with open(csv_path, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([x, y])
    xs = []
    ys = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
        for x, y in data:
            x = int(x)
            y = float(y)
            if (len(xs) > 0 and x == xs[-1]):
                xs.pop(-1)
                ys.pop(-1)
            xs.append(x)
            ys.append(y)
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set(xlabel='Iteration', ylabel='Loss', title='plot_name')
    ax.grid()
    fig.savefig(figure_path)
    plt.close()
