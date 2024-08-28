import jax
import networkx as nx
import numpy as np
import sympy

from .prob import ConvEfn, FuseEfn, Prob


class IsoConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_exprs(self):
        spins = np.array(sympy.symbols(f"s:{self.n*self.n}")).reshape((self.n, self.n))
        g1_edges = np.array(sympy.symbols(f"a:{self.n*(self.n-1)//2}"))
        g2_edges = np.array(sympy.symbols(f"b:{self.n*(self.n-1)//2}"))

        cost_expr = 0
        g1_nedges = 1 - g1_edges
        g2_nedges = 1 - g2_edges
        print(g1_edges)
        print(g2_edges)
        print(g1_edges * g2_nedges[:, np.newaxis])
        print(g2_edges * g1_nedges[:, np.newaxis])

        print(spins * spins[:, np.newaxis])
        g1_once = ((1 - spins.sum(axis=0)) ** 2).sum()
        g2_once = ((1 - spins.sum(axis=1)) ** 2).sum()
        invalid_expr = (self.n) * (sympy.Add(g1_once, g2_once))

        energy_expr = invalid_expr + cost_expr

        self.weights = weights
        return energy_expr, spins.flatten()

    def compile(self, inst):
        sub_dict = {weight: inst_w for weight, inst_w in zip(self.weights, inst)}
        return super().compile(sub_dict)


class IsoFuseEfn(ConvEfn):
    def __init__(self, n):
        self.n = n

    def energy(self):
        pass


class Iso(Prob):
    def __init__(self, args):
        self.n = args.size
        self.nu = args.connectivity
        self.conv_efn = IsoConvEfn(self.n)
        # self.fuse_efn = CutFuseEfn(self.n)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        key, *keys = jax.random.split(key, num=3)

        edges = np.asarray(
            jax.random.bernoulli(keys[0], p=self.nu, shape=combos)
        ).astype(int)
        edges_perm = np.asarray(jax.random.permutation(keys[1], edges))
        return edges, edges_perm

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("iso", help="Graph Isomorphism Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "iso"
