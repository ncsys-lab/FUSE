import argparse
import json
import shutil
import subprocess

from problems.tsp import Tsp


def synthesize(Prob, args):
    prob = Prob(args)
    efn_str = "enc" if args.enc else "conv"
    tag = f"{args.problem}_n{args.size}_{efn_str}"
    synth_dir = f"synths/{tag}"
    template_dir = f"synths/.template_{efn_str}"
    try:
        print("[synth] Running synthesis...")
        shutil.copytree(template_dir, synth_dir, dirs_exist_ok=args.overwrite)
        efn = prob.efn
        efn.gen_circuit(synth_dir)
        # call openlane
        openlane_cmd = (
            f"nix-shell --command 'openlane --run-tag {tag} ../{synth_dir}/config.yaml'"
        )

        subprocess.run(openlane_cmd, shell=True, check=True, cwd="openlane2")
    except FileExistsError:
        print("[synth] Synth dir exists and overwrite is false! Not rerunning synth...")

    # extract metrics
    with open(f"{synth_dir}/runs/{tag}/final/metrics.json", "r") as f:
        metrics = json.load(f)
        area = metrics["design__instance__area"]
        lat = metrics["timing__setup__wns"]

        print("\n====SYNTHESIS RESULTS====")
        print(f"Area (um^2): {area:0.2f}")
        print(f"Latency (ns): {-lat:0.2f}")


def parse(inparser, subparser):
    probs = [Tsp]
    prob_parsers = {prob.gen_parser(subparser): prob for prob in probs}
    args = inparser.parse_args()
    if args.problem not in prob_parsers:
        raise Exception("Problem <%s> not found!" % args.problem)
    return prob_parsers[args.problem], args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Synthesizer")
    parser.add_argument(
        "-f", "--enc", action="store_true", help="Use Encoded Energy Function"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing directory, if it exists",
    )

    subparsers = parser.add_subparsers(
        dest="problem", help="NP Complete Problem to target"
    )
    subparsers.required = True
    prob, args = parse(parser, subparsers)
    res = synthesize(prob, args)
