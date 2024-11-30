import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from circuit import PermuteNet

from .prob import ConvEfn, EncEfn, Prob


class TspConvEfn(ConvEfn):
    def __init__(self, n, maxval):
        self.n = n
        self.maxval = maxval
        self.pbits = 4 * n - 3
        super().__init__()

    def _gen_funcs(self):
        def valid_fn(spins, inst):
            spins = spins.reshape(self.n, self.n)
            location_once = ((1 - spins.sum(axis=0)) ** 2).sum()
            time_once = ((1 - spins.sum(axis=1)) ** 2).sum()
            adjust_derivs = 2 * (spins * (spins - 1)).sum()
            return (self.n * self.maxval) * (location_once + time_once - adjust_derivs)

        def cost_fn(spins, inst):
            spins = spins.reshape(self.n, self.n)
            M = spins.T
            M_p = jnp.roll(M, -1, axis=1).T

            X = M @ M_p
            V = (X + X.T)[jnp.triu_indices_from(X, k=1)].flatten()
            return jnp.dot(V, inst)

        return valid_fn, cost_fn, self.n * self.n


class TspEncEfn(EncEfn):
    def __init__(self, n):
        self.n = n
        net = PermuteNet(n)
        self.permutefn = net.circuitfn()
        self.spins = net.n_spins

        @jax.jit
        def circuitfn(state):
            spins = self.permutefn(state)
            M = spins.T
            M_p = jnp.roll(M, -1, axis=1).T

            X = M @ M_p
            V = (X + X.T)[jnp.triu_indices_from(X, k=1)].flatten()
            return V

        self.circuitfn = circuitfn

    def compile(self, inst):
        return super().compile(inst, self.circuitfn)


class Tsp(Prob):
    def __init__(self, args):
        self.n = args.size
        self.minval = args.minval
        self.maxval = args.maxval
        if args.enc:
            self.efn = TspEncEfn(self.n)
        else:
            self.efn = TspConvEfn(self.n, self.maxval)

    def gen_inst(self, key):
        combos = self.n * (self.n - 1) // 2
        return np.asarray(
            jax.random.randint(
                key, shape=combos, minval=self.minval, maxval=self.maxval
            )
        ).astype(int)

    def sol_inst(self, prob_inst):
        g = nx.Graph()
        for i in range(self.n):
            g.add_node(i)

        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                g.add_edge(i, j, weight=prob_inst[idx])
                idx += 1

        """
        adj_mat = np.zeros(shape=(self.n, self.n))
        adj_mat[np.triu_indices_from(adj_mat, k=1)] = prob_inst
        adj_mat += adj_mat.T

        g = nx.from_numpy_array(adj_mat)
        """

        path = nx.approximation.christofides(g)
        # path = nx.approximation.simulated_annealing_tsp(g, "greedy")
        weight = sum(g[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(path))
        return weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("tsp", help="Travelling Salesman Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-minval", type=int, default=1)
        parser.add_argument("-maxval", type=int, default=10)
        return "tsp"
