import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import sympy

from circuit import PermuteNet

from .prob import ConvEfn, FuseEfn, Prob


class IsoConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_exprs(self):
        combos = self.n * self.n
        spins = np.array(sympy.symbols(f"s:{combos}")).reshape((self.n, self.n))
        g1_mat = np.array(sympy.symbols(f"a:{combos}")).reshape((self.n, self.n))
        g2_mat = np.array(sympy.symbols(f"b:{combos}")).reshape((self.n, self.n))

        g2_p = spins @ g1_mat @ spins.T
        print(g2_mat)
        print(g2_p)
        cost_expr = ((g2_mat) * (1 - g2_p) + g2_p * (1 - g2_mat)).sum()

        g1_once = ((1 - spins.sum(axis=0)) ** 2).sum()
        g2_once = ((1 - spins.sum(axis=1)) ** 2).sum()
        invalid_expr = (self.n) * (sympy.Add(g1_once, g2_once))

        energy_expr = invalid_expr + cost_expr

        self.g1 = g1_mat.flatten()
        self.g2 = g2_mat.flatten()
        return energy_expr, spins.flatten()

    def compile(self, inst):
        g1, g2 = inst
        g1_dict = {edge_v: edge for edge_v, edge in zip(self.g1, g1.flatten())}
        g2_dict = {edge_v: edge for edge_v, edge in zip(self.g2, g2.flatten())}
        sub_dict = {**g1_dict, **g2_dict}

        return super().compile(sub_dict)


class IsoFuseEfn(FuseEfn):
    def __init__(self, n):
        self.n = n
        self.spins, self.permutefn = PermuteNet(self.n)

    def compile(self, inst):
        g1_mat, g2_mat = inst

        @jax.jit
        def circuitfn(state):
            p = self.permutefn(state)
            g2_p = p @ g1_mat @ p.T
            diff = g2_p * (1 - g2_mat) + (1 - g2_p) * (g2_mat)
            return diff.sum()

        weights = 1
        return super().compile(weights, circuitfn)


class Iso(Prob):
    def __init__(self, args):
        self.n = args.size
        self.nu = args.connectivity
        if args.fuse:
            self.efn = IsoFuseEfn(self.n)
        else:
            self.efn = IsoConvEfn(self.n)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        keys = jax.random.split(key, num=3)

        edges = np.asarray(
            jax.random.bernoulli(keys[0], p=self.nu, shape=combos)
        ).astype(int)

        g1_mat = np.zeros((self.n, self.n), dtype=int)
        g1_mat[np.triu_indices(self.n, k=1)] = edges
        g1_mat += g1_mat.T

        perm_mat = jax.random.permutation(keys[1], jnp.eye(self.n, dtype=int))
        g2_mat = perm_mat @ g1_mat @ perm_mat.T
        print(perm_mat)
        print(g1_mat)
        print(g2_mat)
        return g1_mat, g2_mat

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("iso", help="Graph Isomorphism Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.2)
        return "iso"
