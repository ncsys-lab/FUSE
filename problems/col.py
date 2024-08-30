from math import floor, log2

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import sympy

from .prob import ConvEfn, FuseEfn, Prob


class ColConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        self.chi = []
        super().__init__()

    def _gen_exprs(self):
        spins = np.array(sympy.symbols(f"s:{self.n*self.n}")).reshape((self.n, self.n))
        chi = np.array(sympy.symbols(f"x:{self.n}"))
        weights = np.array(sympy.symbols(f"w:{self.n*(self.n-1)//2}"))

        idx1, idx2 = np.triu_indices_from(spins, k=1)
        cost_expr = sympy.Add(
            *(weights[:, np.newaxis] * spins[idx1] * spins[idx2]).flatten()
        )

        invalid_expr = (self.n) * ((1 - (spins * chi).sum(axis=-1)) ** 2).sum()
        energy_expr = invalid_expr + cost_expr

        self.weights = weights
        self.chi = chi

        return energy_expr, spins.flatten()

    def compile(self, inst):
        weights, chi = inst
        weight_dict = {weight: inst_w for weight, inst_w in zip(self.weights, weights)}
        chi_dict = {self.chi[j]: 1 if (j < chi) else 0 for j in range(self.n)}
        sub_dict = {**weight_dict, **chi_dict}
        return super().compile(sub_dict)


class ColFuseEfn(FuseEfn):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.spins = 0

    def compile(self, inst):
        weights, chi = inst

        m = floor(log2(chi))
        vals = [1 << i for i in range(m)]
        vals.append(chi - (1 << m))
        vals = np.array(vals)

        self.spins = self.n * (m + 1)

        @jax.jit
        def circuitfn(spins):
            spins = spins.reshape((self.n, m + 1))
            cols = spins @ vals

            col_mat = jnp.zeros((self.n, chi))
            col_mat = col_mat.at[cols].set(1)
            idx1, idx2 = jnp.triu_indices(self.n, k=1)
            return (col_mat[idx1] * col_mat[idx2]).sum(axis=-1)

        return super().compile(weights, circuitfn)
        # return super().compile(weights, lambda x: circuit(x, self.n, chi, m, vals))


class Col(Prob):
    def __init__(self, args):
        self.n = args.size
        self.nu = args.connectivity
        if args.fuse:
            self.efn = ColFuseEfn(self.n)
        else:
            self.efn = ColConvEfn(self.n)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        weights = np.asarray(jax.random.bernoulli(key, p=self.nu, shape=combos)).astype(
            int
        )

        # Solve the instance here in order to give upper bound on chi for compilation
        adj_mat = np.zeros(shape=(self.n, self.n))
        adj_mat[np.triu_indices_from(adj_mat, k=1)] = weights
        g = nx.from_numpy_array(adj_mat)
        color = nx.greedy_color(g)
        chi = max(color.values()) + 1
        return weights, chi

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("col", help="Graph Coloring Problem")
        parser.add_argument("-n", "--size", type=int, default=15)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "col"
