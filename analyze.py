import argparse

from log import Logger
from util import print_run_stats


def analyze_run(args):
    ctss = []
    sol_quals = []
    sol_cycs = []
    succs = 0
    succs_10 = 0
    for log in args.logs:
        log_data = Logger.load_log(log)
        ctss.append(log_data["cts"])
        sol_quals.append(log_data["sol_qual"])
        sol_cycs.append(log_data["sol_cyc"])
        succs += log_data["succ"]
        succs_10 += log_data["succ"]

    trials = len(args.logs)
    esp = succs / trials
    esp_10 = succs_10 / trials
    print_run_stats(args.iters, esp, esp_10, ctss, sol_quals, sol_cycs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Visualizer")
    parser.add_argument("logs", type=str, nargs="+", help="Logfiles to analyze")
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        help="(Maximum) number of iterations",
        default=1000000,
    )
    args = parser.parse_args()
    analyze_run(args)
