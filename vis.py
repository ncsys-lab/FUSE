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

        """
        # USE FOR CONV FIGURE (1.A)

        axins = ax.inset_axes(
            [0.35, 0.35, 0.60, 0.60], xlim=(-100, 10000), ylim=(0, 4000)
        )

        axins.scatter(
            idx[filter_idx],
            energies[filter_idx],
            color="gray",
            s=4,
            linewidth=0,
            zorder=1,
        )
        axins.scatter(
            idx[filter_idx][valid_filter],
            energies[filter_idx][valid_filter],
            label="valid",
            color="tab:blue",
            linewidth=0,
            s=12,
            zorder=2,
        )
        ax.indicate_inset_zoom(axins, edgecolor="black")
        ax.set_ylim(-1000, 18000)
        """

        """
        # USE FOR ENC FIGURE (1.B)

        """
        axins = ax.inset_axes([0.25, 0.25, 0.70, 0.70], xlim=(-50, 500), ylim=(-1, 80))

        axins.scatter(
            idx[filter_idx],
            energies[filter_idx],
            color="gray",
            s=4,
            linewidth=0,
            zorder=1,
        )
        axins.scatter(
            idx[filter_idx][valid_filter],
            energies[filter_idx][valid_filter],
            label="valid",
            color="tab:blue",
            linewidth=0,
            s=12,
            zorder=2,
        )
        # ax.indicate_inset_zoom(axins, edgecolor="black")
        ax.set_ylim(-1000, 18000)

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
    args = parser.parse_args()
    gen_vis(args)
