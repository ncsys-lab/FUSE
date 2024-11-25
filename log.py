import os

import jax
import jax.numpy as jnp


class Logger:
    def __init__(self, args):
        efn_str = "enc" if args.enc else "conv"
        beta_str = f"b{args.beta_init:0.2f}_{args.beta_end:0.2f}_{'log' if args.beta_log else 'lin'}"
        log_dir = f"log/{args.problem}_n{args.size}_{efn_str}_{beta_str}_s{args.seed}"
        exist_ok = args.overwrite if args.trials else True
        os.makedirs(log_dir, exist_ok=exist_ok)
        self.log_dir = log_dir
        self.log_idx = (jnp.arange(args.iters * args.log_rate) / args.log_rate).astype(
            int
        )

    def log(self, run_key, res):
        prob_sol, succ, cts, sol_qual, trace = res
        if trace is not None:
            states, energies = trace
            states = states[self.log_idx]
            energies = energies[self.log_idx]
        else:
            states = []
            energies = []

        key0, key1 = jax.random.key_data(run_key)
        log_file = f"{self.log_dir}/0x{key0:08X}_{key1:08X}.log"
        jnp.savez(
            log_file,
            idx=self.log_idx,
            prob_sol=prob_sol,
            succ=succ[0],
            succ_10=succ[1],
            cts=cts,
            sol_qual=sol_qual[0],
            sol_cyc=sol_qual[1],
            states=states,
            energies=energies,
        )

    @staticmethod
    def load_log(log_file):
        return jnp.load(log_file)
