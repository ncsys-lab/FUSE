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


def run(key, quick, iters, prob, efn, beta_i, betafn):
    key, prob_key = jax.random.split(key)
    prob_inst = prob.gen_inst(prob_key)
    prob_sol = prob.sol_inst(prob_inst)

    print("[lower] Lowering to p-computer...")
    start_time = time.perf_counter()
    engradfn, masks = efn.compile(prob_inst)
    runtime = time.perf_counter() - start_time
    print(f"[lower] Lowering time was {runtime:0.2f}")

    print("[run] Beginning execution...")
    start_time = time.perf_counter()
    n_masks = len(masks)
    p = masks[0].shape

    key, state_key = jax.random.split(key)
    state = jax.random.bernoulli(state_key, shape=p).astype(int)

    key, run_key = jax.random.split(key)

    x = jnp.arange(iters)

    def inner_loop(key, i, beta, state):
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
        return key, beta, new_state, energy

    if quick:

        def while_inner(val):
            (i, key, beta, state, energy) = val
            key, beta, new_state, energy = inner_loop(key, i, beta, state)
            return (i + 1, key, beta, new_state, energy)

        def cond_fun(val):
            (i, _, _, _, energy) = val
            return (i == 0) + (energy > prob_sol) * (i < iters)

        init_pargs = (jnp.int64(0), run_key, beta_i, state, jnp.float64(0.0))
        (i, _, _, state, energy) = jax.lax.while_loop(cond_fun, while_inner, init_pargs)
        min_energy = energy
        sol_cyc = i
        succ = energy <= prob_sol
        succ_10 = i < iters // 10
        trace = None
        cts = i if succ else jnp.array([-1])

    else:

        def for_inner(pargs, i):
            (key, beta, state) = pargs
            key, beta, new_state, energy = inner_loop(key, i, beta, state)
            return (key, beta, new_state), (state, energy)

        init_pargs = (run_key, beta_i, state)
        _, trace = jax.lax.scan(for_inner, init_pargs, x, iters)

        _, energies = trace
        min_energy = jnp.min(energies)
        sol_cyc = jnp.argmin(energies)
        gated = energies <= prob_sol
        succ = jnp.sum(gated) > 0
        succ_10 = jnp.sum(gated[: iters // 10]) > 0
        cts = jnp.argmax(gated) if succ else jnp.array(-1)

    sol_qual = (
        (prob_sol - min_energy) / abs(prob_sol) if prob_sol != 0 else jnp.array(-1)
    )

    runtime = time.perf_counter() - start_time
    print(f"[run] Done! Runtime was {runtime:0.2f}")
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
        res = run(run_key, args.quick, args.iters, prob, efn, args.beta_init, betafn)
        _, succ, cts, sol_qual, _ = res

        logger.log(run_key, res)
        start_time = time.perf_counter()

        print("==== RUN STATS ====")
        print(f"CtS: {cts.item():.0f}")
        print(f"Best Cycle: {sol_qual[1]:.0f}")
        print(f"Sol qual(%): {sol_qual[0]:0.02f}")

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
                        quick=args.quick,
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
            logger.log(run_key, res)

        esp = succs / args.trials
        esp_10 = succs_10 / args.trials
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
        "-q",
        "--quick",
        action="store_true",
        help="Enable quick execution via early exiting",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        help="(Maximum) number of iterations",
        default=1000000,
    )
    parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
    parser.add_argument(
        "-f", "--enc", action="store_true", help="Use Encoded Energy Function"
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
    subparsers.required = True
    prob, args = parse(parser, subparsers)
    res = execute(prob, args)