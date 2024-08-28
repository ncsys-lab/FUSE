import jax
import networkx as nx
import numpy as np
import sympy

from .prob import ConvEfn, FuseEfn, Prob


class TspConvEfn(ConvEfn):
    def __init__(self, n, maxval):
        self.n = n
        self.maxval = maxval
        super().__init__()

    def _gen_exprs(self):
        spins = np.array(sympy.symbols(f"s:{self.n*self.n}")).reshape((self.n, self.n))
        weights = np.array(sympy.symbols(f"w:{self.n*(self.n-1)//2}"))

        M = spins.T
        M_p = np.roll(M, -1, axis=1).T

        X = M @ M_p
        V = (
            X[np.triu_indices_from(X, k=1)].flatten()
            + X[np.tril_indices_from(X, k=-1)].flatten()
        )
        cost_expr = np.dot(V, weights)
        location_once = ((1 - spins.sum(axis=0)) ** 2).sum()
        time_once = ((1 - spins.sum(axis=1)) ** 2).sum()
        invalid_expr = (self.n * self.maxval) * (sympy.Add(location_once, time_once))

        energy_expr = invalid_expr + cost_expr

        self.weights = weights
        return energy_expr, spins.flatten()

    def compile(self, inst):
        sub_dict = {weight: inst_w for weight, inst_w in zip(self.weights, inst)}
        return super().compile(sub_dict)


class TspFuseEfn(ConvEfn):
    def __init__(self, n):
        self.n = n

    def energy(self):
        pass


class Tsp(Prob):
    def __init__(self, args):
        self.n = args.size
        self.minval = args.minval
        self.maxval = args.maxval
        self.conv_efn = TspConvEfn(self.n, self.maxval)
        # self.fuse_efn = CutFuseEfn(self.n)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        return np.asarray(
            jax.random.uniform(
                key, shape=combos, minval=self.minval, maxval=self.maxval
            )
        ).astype(int)

    def sol_inst(self, prob_inst):
        adj_mat = np.zeros(shape=(self.n, self.n))
        adj_mat[np.triu_indices_from(adj_mat, k=1)] = prob_inst
        g = nx.from_numpy_array(adj_mat)
        path = nx.approximation.traveling_salesman_problem(g)
        weight = sum(g[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(path))
        return weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("tsp", help="Travelling Salesman Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-minval", type=int, default=1)
        parser.add_argument("-maxval", type=int, default=10)
        return "tsp"
