import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from log import Logger
from problems.col import Col
from problems.cut import Cut
from problems.iso import Iso
from problems.knp import Knp
from problems.stp import Stp
from problems.tsp import Tsp


def gen_vis(args):
    for log in args.logs:
        log_data = Logger.load_log(log)
        idx = log_data["idx"]
        energies = log_data["energies"]
        if log_data["succ"]:
            filter_idx = idx < log_data["cts"]
        else:
            filter_idx = jnp.ones_like(idx)

        plt.scatter(
            idx[filter_idx],
            energies[filter_idx],
            color="gray",
            s=4,
            linewidth=0,
            zorder=1,
        )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Visualizer")
    parser.add_argument("logs", type=str, nargs="+", help="logfiles")
    args = parser.parse_args()
    gen_vis(args)
