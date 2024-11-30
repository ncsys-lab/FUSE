import jax
import jax.numpy as jnp
import numpy as np

from circuit import PermuteNet

from .prob import ConvEfn, EncEfn, Prob


class IsoConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_funcs(self):
        def valid_fn(spins, inst):
            spins = spins.reshape(self.n, self.n)
            g1_once = ((1 - spins.sum(axis=0)) ** 2).sum()
            g2_once = ((1 - spins.sum(axis=1)) ** 2).sum()
            adjust_derivs = 2 * (spins * (spins - 1)).sum()
            return (self.n) * (g1_once + g2_once - adjust_derivs)

        def cost_fn(spins, inst):
            spins = spins.reshape(self.n, self.n)
            g1_mat, g2_mat = inst
            g2_p = spins @ g1_mat @ spins.T
            return ((g2_mat) * (1 - g2_p) + g2_p * (1 - g2_mat)).sum()

        return valid_fn, cost_fn, self.n * self.n


class IsoEncEfn(EncEfn):
    def __init__(self, n):
        super().__init__()
        self.n = n
        net = PermuteNet(n)
        self.permutefn = net.circuitfn()
        self.spins = net.n_spins

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
        if args.enc:
            self.efn = IsoEncEfn(self.n)
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
        return g1_mat, g2_mat

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("iso", help="Graph Isomorphism Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "iso"
