import argparse
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import multiprocess.context as ctx
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from log import Logger
from problems.col import Col
from problems.cut import Cut
from problems.iso import Iso
from problems.knp import Knp
from problems.stp import Stp
from problems.tsp import Tsp
from util import print_run_stats

ctx._force_start_method("spawn")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
DEBUG = False


def mute():
    sys.stdout = open(os.devnull, "w")
    pass


# @partial(jax.jit, static_argnums=(4, 5, 6))
def run(key, iters, prob, efn, beta_i, betafn):
    key, prob_key = jax.random.split(key)
    prob_inst = prob.gen_inst(prob_key)
    prob_sol = prob.sol_inst(prob_inst)

    print("[lower] Lowering!")
    start_time = time.perf_counter()
    engradfn, masks = efn.compile(prob_inst)
    runtime = time.perf_counter() - start_time
    print(f"[lower] Lowering was {runtime:0.2f}")

    print("[run] Running!")
    start_time = time.perf_counter()
    n_masks = len(masks)
    p = masks[0].shape

    def inner_loop(pargs, i):
        (key, beta, state) = pargs
        key, subkey = jax.random.split(key)
        mask = masks[i % n_masks]
        energy, grad = engradfn(state, mask)
        if DEBUG:
            jax.debug.print(
                "{i}:\tEnergy: {energy}\t, State: {state}",
                i=i,
                energy=energy,
                state=jnp.astype(state, int),
            )
        rand = jax.random.uniform(subkey, shape=p)
        pos = (jax.nn.sigmoid(-beta * grad) - rand > 0).flatten()
        new_state = jnp.where(mask, pos, state)
        beta = betafn(beta)
        return (key, beta, new_state), (state, energy)

    key, subkey = jax.random.split(key)
    state = jax.random.bernoulli(subkey, shape=p).astype(int)

    key, subkey = jax.random.split(key)
    init_pargs = (subkey, beta_i, state)

    x = jnp.arange(iters)
    _, trace = jax.lax.scan(inner_loop, init_pargs, x, iters)

    bits, energies = trace
    # print(bits[jnp.argmin(energies)])
    # print(jnp.min(energies))
    # print(jnp.unique(bits, axis=0).shape)

    min_energy = jnp.min(energies)
    sol_cyc = jnp.argmin(energies)
    sol_qual = (
        (prob_sol - min_energy) / abs(prob_sol) if prob_sol != 0 else jnp.array(-1)
    )

    gated = energies <= prob_sol
    succ = jnp.sum(gated) > 0

    succ_10 = jnp.sum(gated[: iters // 10]) > 0

    cts = jnp.argmax(gated) if succ else jnp.array(-1)

    runtime = time.perf_counter() - start_time
    print(f"[run] Runtime was {runtime:0.2f}")
    return (
        prob_sol,
        (succ, succ_10),
        cts,
        (sol_qual, sol_cyc),
        trace,
    )


def execute(Prob, args):
    key = jax.random.key(args.seed)
    start_time = time.perf_counter()
    prob = Prob(args)
    runtime = time.perf_counter() - start_time
    print(f"[compile] Compile time was {runtime:0.2f}")
    efn = prob.efn

    logger = Logger(args)

    @jax.jit
    def betafn(beta: float) -> float:
        if args.beta_log:
            raise NotImplementedError
        else:
            return beta + (10**args.beta_end - args.beta_init) / args.iters

    if args.trials is None:
        key, run_key = jax.random.split(key)
        res = run(run_key, args.iters, prob, efn, args.beta_init, betafn)
        print(res)
        _, succ, cts, sol_qual, trace = res

        bits, energy = trace
        # perm_out = efn.permutefn(bits[jnp.argmin(energy)])
        # print(perm_out.astype(int))

        logger.log(run_key, res)
        start_time = time.perf_counter()

    else:
        print("Batch mode! Will not print out hints...")

        run_keys = jax.random.split(key, num=args.trials)
        p = Pool(args.threads, initializer=mute)

        ctss = []
        sol_quals = []
        sol_cycs = []
        succs = 0
        succs_10 = 0
        for res, run_key in zip(
            tqdm(
                p.imap(
                    partial(
                        run,
                        iters=args.iters,
                        prob=prob,
                        efn=efn,
                        beta_i=args.beta_init,
                        betafn=betafn,
                    ),
                    run_keys,
                ),
                total=args.trials,
            ),
            run_keys,
        ):
            _, succ, cts, sol_qual, _ = res
            ctss.append(cts.item())
            sol_quals.append(sol_qual[0].item())
            sol_cycs.append(sol_qual[1].item())
            succs += succ[0]
            succs_10 += succ[1]
            # logger.log(run_key, res)
        esp = succs / args.trials
        esp_10 = succs_10 / args.trials
        # print(f"Success: {esp:0.3f}")
        print_run_stats(esp, esp_10, ctss, sol_quals, sol_cycs)


def parse(inparser, subparser):
    probs = [Cut, Col, Tsp, Iso, Knp, Stp]
    prob_parsers = {prob.gen_parser(subparser): prob for prob in probs}
    args = inparser.parse_args()
    if args.problem not in prob_parsers:
        raise Exception("unknown subparser <%s>" % args.problem)
    return prob_parsers[args.problem], args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Simulator")
    parser.add_argument("-t", "--trials", type=int, help="Number of trials")
    parser.add_argument(
        "-x", "--threads", type=int, default=6, help="Number of threads to use"
    )
    parser.add_argument(
        "-i", "--iters", type=int, help="Number of iterations", default=1000000
    )
    parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
    parser.add_argument(
        "-f", "--fuse", action="store_true", help="Use FUSE Energy Function"
    )
    parser.add_argument(
        "-bi", "--beta_init", type=float, default=0.0, help="Initial Beta value"
    )
    parser.add_argument(
        "-be", "--beta_end", type=float, default=1.0, help="Ending Beta value"
    )
    parser.add_argument(
        "-bl", "--beta_log", action="store_true", help="Use Logarithmic Beta scaling"
    )
    parser.add_argument(
        "-lr",
        "--log_rate",
        default=0.001,
        help="Proportion of logs to keep",
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
    prob, args = parse(parser, subparsers)
    res = execute(prob, args)
