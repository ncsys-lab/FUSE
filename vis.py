import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt

from log import Logger


def gen_vis(args):
    for log in args.logs:
        log_data = Logger.load_log(log)
        plot_file = Logger.get_plot_file(log)
        idx = log_data["idx"]
        energies = log_data["energies"]
        valids = log_data["valids"]
        if log_data["succ"]:
            filter_idx = idx <= log_data["cts"]
        else:
            filter_idx = jnp.ones_like(idx, dtype=bool)

        valid_filter = valids[filter_idx]

        _, ax1 = plt.subplots()

        ax1.scatter(
            idx[filter_idx],
            energies[filter_idx],
            color="gray",
            s=4,
            linewidth=0,
            zorder=1,
        )
        ax1.scatter(
            idx[filter_idx][valid_filter],
            energies[filter_idx][valid_filter],
            label="valid",
            color="tab:blue",
            linewidth=0,
            s=12,
            zorder=2,
        )

        # ax1.set_ylim(-100, 4000)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Energy")
        plt.savefig(plot_file.with_suffix("." + args.filetype), dpi=400)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Visualizer")
    parser.add_argument("logs", type=str, nargs="+", help="Logfiles to analyze")
    parser.add_argument(
        "-ft", "--filetype", type=str, default="png", help="Filetype for plot"
    )
    args = parser.parse_args()
    gen_vis(args)
