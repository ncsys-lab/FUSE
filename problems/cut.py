import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from .prob import ConvEfn, Prob


class CutConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_funcs(self):
        def valid_fn(spins, inst):
            return 0.0

        def cost_fn(spins, inst):
            spins = spins[:, jnp.newaxis]
            n_spins = 1 - spins
            edge_mat = spins @ n_spins.T + n_spins @ spins.T
            return jnp.dot(
                -inst, edge_mat[jnp.triu_indices_from(edge_mat, k=1)].flatten()
            )

        return valid_fn, cost_fn, self.n


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
        adj_mat += adj_mat.T
        g = nx.from_numpy_array(adj_mat)
        weight, assign = nx.approximation.one_exchange(g)
        print(assign)
        print(adj_mat)
        print(weight)
        return -weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("cut", help="Max-Cut Problem")
        parser.add_argument("-n", "--size", type=int, default=15)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "cut"
