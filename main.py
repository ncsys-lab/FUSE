import argparse
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
from problems.tsp import Tsp
from util import print_cts_stats

ctx._force_start_method("spawn")
jax.config.update("jax_platform_name", "cpu")
DEBUG = False


def mute():
    pass


# @partial(jax.jit, static_argnums=(4, 5, 6))
def run(key, iters, prob, efn, beta_i, betafn):
    key, prob_key = jax.random.split(key)
    prob_inst = prob.gen_inst(prob_key)
    prob_sol = prob.sol_inst(prob_inst)
    energyfn, gradfn, masks = efn.compile(prob_inst)

    n_masks = len(masks)
    p = masks[0].shape

    def inner_loop(pargs, i):
        (key, beta, state) = pargs
        key, subkey = jax.random.split(key)

        grad = gradfn(state)
        energy = energyfn(state)
        if DEBUG:
            jax.debug.print(
                "{i}:\tEnergy: {energy}\t, State: {state}",
                i=i,
                energy=energy,
                state=jnp.astype(state, int),
            )
        rand = jax.random.uniform(subkey)
        pos = (jax.nn.sigmoid(-beta * grad) - rand > 0).flatten()
        curr_mask = masks[i % n_masks]
        new_state = jnp.where(curr_mask, pos, state)
        beta = betafn(beta)
        return (key, beta, new_state), (new_state, energy)

    key, subkey = jax.random.split(key)
    state = jax.random.bernoulli(subkey, shape=p).astype(int)

    key, subkey = jax.random.split(key)
    init_pargs = (subkey, beta_i, state)

    x = jnp.arange(iters)
    _, trace = jax.lax.scan(inner_loop, init_pargs, x, iters)

    bits, energies = trace
    gated = energies <= prob_sol
    succ = jnp.sum(gated) > 0
    cts = jnp.argmax(gated) if succ else -1
    return (
        prob_sol,
        succ,
        cts,
        trace,
    )


def execute(Prob, args):
    key = jax.random.key(args.seed)
    prob = Prob(args)
    efn = prob.fuse_efn if args.fuse else prob.conv_efn

    logger = Logger(args)

    @jax.jit
    def betafn(beta: float) -> float:
        if args.beta_log:
            raise NotImplementedError
        else:
            return beta + (args.beta_end - args.beta_init) / args.iters

    if args.trials is None:
        print("[run] Running!")
        start_time = time.perf_counter()
        key, run_key = jax.random.split(key)
        res = run(run_key, args.iters, prob, efn, args.beta_init, betafn)
        print(res)
        logger.log(run_key, res)
        runtime = time.perf_counter() - start_time
        print(f"[run] Runtime was {runtime:0.2f}")

    else:
        print("Batch mode! Will not print out hints...")

        key, *run_keys = jax.random.split(key, num=args.trials)
        p = Pool(6, initializer=mute)

        ctss = []
        succs = 0
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
            _, succ, cts, _ = res
            ctss.append(cts.item())
            succs += succ
            # logger.log(run_key, res)
        print_cts_stats(ctss)
        print(f"Success: {succs/args.trials:0.3f}")


def parse(inparser, subparser):
    probs = [Cut, Col, Tsp, Iso]
    prob_parsers = {prob.gen_parser(subparser): prob for prob in probs}
    args = inparser.parse_args()
    if args.problem not in prob_parsers:
        raise Exception("unknown subparser <%s>" % args.problem)
    return prob_parsers[args.problem], args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUSE Simulator")
    parser.add_argument("-t", "--trials", type=int, help="Number of trials")
    parser.add_argument(
        "-i", "--iters", type=int, help="Number of iterations", default=1000000
    )
    parser.add_argument("-s", "--seed", type=int, help="Random Seed", default=0)
    parser.add_argument(
        "-f", "--fuse", action="store_true", help="Use FUSE Energy Function"
    )
    parser.add_argument("-bi", "--beta_init", default=0.0, help="Initial Beta value")
    parser.add_argument("-be", "--beta_end", default=1.0, help="Ending Beta value")
    parser.add_argument(
        "-bl", "--beta_log", action="store_true", help="Use Logarithmic Beta scaling"
    )
    parser.add_argument(
        "-lr",
        "--log_rate",
        default=0.01,
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
