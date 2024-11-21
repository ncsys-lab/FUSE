import jax
import networkx as nx
import numpy as np
import symengine as se

from .prob import ConvEfn, Prob


class CutConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_exprs(self):
        spins = np.array(se.symbols(f"s:{self.n}"))[:, np.newaxis]
        weights = np.array(se.symbols(f"w:{self.n*(self.n-1)//2}"))

        n_spins = 1 - spins
        edge_mat = spins @ n_spins.T + n_spins @ spins.T
        energy_expr = np.dot(
            -weights, edge_mat[np.triu_indices_from(edge_mat, k=1)].flatten()
        )

        self.weights = weights
        return energy_expr, spins.flatten()

    def compile(self, inst):
        sub_dict = {weight: inst_w for weight, inst_w in zip(self.weights, inst)}
        return super().compile(sub_dict)


class Cut(Prob):
    def __init__(self, args):
        self.n = args.size
        self.nu = args.connectivity
        if args.enc:
            self.efn = None
        else:
            self.efn = CutConvEfn(self.n)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        return np.asarray(jax.random.bernoulli(key, p=self.nu, shape=combos)).astype(
            int
        )

    def sol_inst(self, prob_inst):
        adj_mat = np.zeros(shape=(self.n, self.n))
        adj_mat[np.triu_indices_from(adj_mat, k=1)] = prob_inst
        g = nx.from_numpy_array(adj_mat)
        weight, _ = nx.approximation.one_exchange(g)
        return -weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("cut", help="Max-Cut Problem")
        parser.add_argument("-n", "--size", type=int, default=15)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "cut"
