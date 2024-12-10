import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt

from log import Logger


def gen_vis(args):
    for log in args.logs:
        log_data = Logger.load_log(log)
        plot_file = Logger.get_plot_file(log, name=args.name)
        idx = log_data["idx"]
        energies = log_data["energies"]
        valids = log_data["valids"]
        if log_data["succ"]:
            filter_idx = idx <= log_data["cts"]
        else:
            filter_idx = jnp.ones_like(idx, dtype=bool)

        valid_filter = valids[filter_idx]

        _, ax = plt.subplots()

        ax.scatter(
            idx[filter_idx],
            energies[filter_idx],
            color="gray",
            s=4,
            linewidth=0,
            zorder=1,
        )
        ax.scatter(
            idx[filter_idx][valid_filter],
            energies[filter_idx][valid_filter],
            label="valid",
            color="tab:blue",
            linewidth=0,
            s=12,
            zorder=2,
        )

        if args.ylimit != []:
            ax.set_ylim(*args.ylimit)

        if args.xlimit != []:
            ax.set_xlim(*args.xlimit)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        plt.savefig(plot_file.with_suffix("." + args.filetype), dpi=400)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Visualizer")
    parser.add_argument("logs", type=str, nargs="+", help="Logfiles to analyze")
    parser.add_argument(
        "-ft", "--filetype", type=str, default="png", help="Filetype for plot"
    )
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="Name for plot file"
    )
    parser.add_argument(
        "-ylim", "--ylimit", type=int, nargs=2, default=[], help="Y-limits for plot"
    )
    parser.add_argument(
        "-xlim", "--xlimit", type=int, nargs=2, default=[], help="X-limits for plot"
    )
    args = parser.parse_args()
    gen_vis(args)
