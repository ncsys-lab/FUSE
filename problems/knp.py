from math import floor, log2

import jax
import jax.numpy as jnp
import knapsack as knap_solver
import numpy as np
import sympy

from .prob import ConvEfn, EncEfn, Prob


class KnpConvEfn(ConvEfn):
    def __init__(self, n, cap, c_maxval):
        self.n = n
        self.cap = cap
        self.c_maxval = c_maxval
        super().__init__()

    # Costs = higher is better
    # Weights = lower is better
    def _gen_exprs(self):
        spins = np.array(sympy.symbols(f"s:{self.n}"))
        costs = np.array(sympy.symbols(f"c:{self.n}"))
        weights = np.array(sympy.symbols(f"w:{self.n}"))

        cap = int(self.cap) + 1
        m = floor(log2(cap))
        vals = [1 << i for i in range(m)]
        vals.append(cap - (1 << m))

        w_spins = np.array(sympy.symbols(f"ws:{len(vals)}"))
        vals = np.array(vals)

        cost_expr = -np.dot(spins, costs)
        weight_expr = np.dot(spins, weights)
        cweight_expr = np.dot(w_spins, vals)

        invalid_expr = self.c_maxval * self.n * (weight_expr - cweight_expr) ** 2
        energy_expr = invalid_expr + cost_expr

        self.costs = costs
        self.weights = weights

        out_spins = np.hstack((spins, w_spins))
        return energy_expr, out_spins

    def compile(self, inst):
        weights, costs = inst
        weight_dict = {weight: inst_w for weight, inst_w in zip(self.weights, weights)}
        cost_dict = {cost: inst_c for cost, inst_c in zip(self.costs, costs)}
        sub_dict = {**weight_dict, **cost_dict}
        return super().compile(sub_dict)


class KnpEncEfn(EncEfn):
    def __init__(self, n, cap, c_maxval):
        super().__init__()
        self.n = n
        self.spins = n
        self.cap = cap
        self.c_maxval = c_maxval

    @staticmethod
    @jax.jit
    def circuit(spins, cap, weights):
        return jnp.append(spins, (spins @ weights > cap).astype(int))

    def compile(self, inst):
        weights, costs = inst
        e_weights = jnp.append(-costs, self.n * self.c_maxval)
        return super().compile(
            e_weights, lambda x: KnpEncEfn.circuit(x, self.cap, weights)
        )


class Knp(Prob):
    def __init__(self, args):
        self.n = args.size
        self.w_maxval = args.w_maxval
        self.w_minval = args.w_minval
        self.c_maxval = args.c_maxval
        self.c_minval = args.c_minval
        self.rel_cap = args.rel_cap
        self.cap = self.n * self.w_maxval * self.rel_cap
        if args.enc:
            self.efn = KnpEncEfn(self.n, self.cap, self.c_maxval)
        else:
            self.efn = KnpConvEfn(self.n, self.cap, self.c_maxval)

    def gen_inst(self, key):
        key, *keys = jax.random.split(key, num=3)
        weights = np.asarray(
            jax.random.randint(
                keys[0],
                shape=(self.n),
                minval=self.w_minval,
                maxval=self.w_maxval,
                dtype=int,
            )
        )
        costs = np.asarray(
            jax.random.randint(
                keys[1],
                shape=(self.n),
                minval=self.c_minval,
                maxval=self.c_maxval,
                dtype=int,
            )
        )
        return (weights, costs)

    def sol_inst(self, prob_inst):
        weights, costs = prob_inst
        weight, result = knap_solver.knapsack(size=weights, weight=costs).solve(
            self.cap
        )
        return -weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("knp", help="Knapsack Problem")
        parser.add_argument("-n", "--size", type=int, default=15)
        parser.add_argument("-w_minval", type=int, default=1)
        parser.add_argument("-w_maxval", type=int, default=10)
        parser.add_argument("-c_minval", type=int, default=1)
        parser.add_argument("-c_maxval", type=int, default=10)
        parser.add_argument("-f", "--rel_cap", type=float, default=0.33)
        return "knp"
