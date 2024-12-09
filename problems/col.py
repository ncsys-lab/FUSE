from math import floor, log2

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from circuit import SelectNet

from .prob import ConvEfn, EncEfn, Prob

FUSE_COLOR = False


class ColConvEfn(ConvEfn):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def _gen_funcs(self):
        def valid_fn(spins, inst):
            spins = spins.reshape((self.n, -1))
            one_color = ((1 - spins.sum(axis=-1)) ** 2).sum()
            adjust_derivs = (spins * (spins - 1)).sum()
            return (self.n) * (one_color - adjust_derivs)

        def cost_fn(spins, inst):
            spins = spins.reshape((self.n, -1))
            idx1, idx2 = jnp.triu_indices(self.n, k=1)
            return (inst[:, jnp.newaxis] * spins[idx1] * spins[idx2]).sum()

        return valid_fn, cost_fn, None

    def compile(self, inst):
        weights, chi, _ = inst
        self.spins = self.n * chi
        return super().compile(weights)


class ColEncEfn(EncEfn):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def compile(self, inst):
        n = self.n

        weights, chi, color_dict = inst

        net = SelectNet(n, chi)
        self.selectfn = net.circuitfn()
        self.spins = net.n_spins

        nspins = chi - 1

        idx1, idx2 = jnp.triu_indices(n, k=1)

        @jax.jit
        def circuitfn(spins):
            col_mat = self.selectfn(spins.reshape(n, nspins))
            return (col_mat[idx1] & col_mat[idx2]).sum(axis=-1)

        if FUSE_COLOR:
            adjs = np.zeros((n, n, nspins, nspins))
            idx1, idx2 = np.triu_indices(n, k=1)
            adjs[idx1, idx2, :, :] = weights[..., np.newaxis, np.newaxis]

            adjs = adjs.swapaxes(1, 2).reshape((n * nspins, n * nspins))
            adjs += adjs.T
            adjs += np.kron(np.eye(n), 1 - np.eye(nspins, dtype=bool))

            color_dict = nx.greedy_color(nx.from_numpy_array(adjs))
            colors = np.fromiter(
                [color_dict[spin] for spin in range(n * nspins)], dtype=int
            )
            ncolors = max(colors) + 1

            masks = np.zeros((ncolors, n * nspins), dtype=np.bool_)
            masks[colors, np.arange(colors.size)] = 1

            masks = jnp.asarray(masks)

            vcircuitfn = jax.vmap(circuitfn)
            return super().compile(
                weights, circuitfn, vcircuitfn=vcircuitfn, masks=masks
            )
        else:
            return super().compile(weights, circuitfn)


class ColLogEncEfn(EncEfn):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def compile(self, inst):
        weights, chi, _ = inst

        m = floor(log2(chi))
        vals = [1 << i for i in range(m)]
        if chi != (1 << m):
            vals.append(chi - (1 << m))

        vals = np.array(vals)
        self.spins = self.n * (vals.shape[0])

        @jax.jit
        def circuitfn(spins):
            spins = spins.reshape((self.n, vals.shape[0]))
            cols = spins @ vals

            col_mat = jnp.zeros((self.n, chi))
            col_mat = col_mat.at[jnp.arange(self.n), cols].set(1)
            idx1, idx2 = jnp.triu_indices(self.n, k=1)
            return (col_mat[idx1] * col_mat[idx2]).sum(axis=-1)

        return super().compile(weights, circuitfn)


class Col(Prob):
    def __init__(self, args):
        self.n = args.size
        self.nu = args.connectivity
        if args.enc:
            if args.logn:
                self.efn = ColLogEncEfn(self.n)
            else:
                self.efn = ColEncEfn(self.n)
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
        adj_mat += adj_mat.T
        g = nx.from_numpy_array(adj_mat)
        color = nx.greedy_color(g, strategy="DSATUR")
        chi = max(color.values()) + 1
        return weights, chi, color

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("col", help="Graph Coloring Problem")
        parser.add_argument("-n", "--size", type=int, default=15)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        parser.add_argument("--logn", action="store_true")
        return "col"
