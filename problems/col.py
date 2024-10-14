from math import floor, log2

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import sympy

from circuit import SelectNet

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
            *(weights[:, np.newaxis] * chi * spins[idx1] * spins[idx2]).flatten()
        )

        invalid_expr = (self.n) * ((1 - (spins * chi).sum(axis=-1)) ** 2).sum()
        print(invalid_expr)
        energy_expr = invalid_expr + cost_expr

        self.weights = weights
        self.chi = chi

        return energy_expr, spins.flatten()

    def compile(self, inst):
        weights, chi, _ = inst
        weight_dict = {weight: inst_w for weight, inst_w in zip(self.weights, weights)}
        chi_dict = {self.chi[j]: 1 if j < chi else 0 for j in range(self.n)}
        sub_dict = {**weight_dict, **chi_dict}
        return super().compile(sub_dict)


class ColFuseEfn(FuseEfn):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def compile(self, inst):
        n = self.n

        weights, chi, color_dict = inst
        self.spins, self.selectfn = SelectNet(n, chi)

        nspins = chi - 1

        @jax.jit
        def circuitfn(spins):
            col_mat = self.selectfn(spins.reshape(n, nspins))
            # print(col_mat)
            # jax.debug.print("spins: {spins}", spins=spins.reshape(self.n, self.n - 1))
            # jax.debug.print("Col_Mat: {col_mat}", col_mat=col_mat)
            idx1, idx2 = jnp.triu_indices(n, k=1)
            return (col_mat[idx1] * col_mat[idx2]).sum(axis=-1)

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

        print(f"colors: {ncolors}")

        masks = np.zeros((ncolors, n * nspins), dtype=np.bool_)
        masks[colors, np.arange(colors.size)] = 1

        masks = jnp.asarray(masks)

        vcircuitfn = jax.vmap(circuitfn)
        return super().compile(weights, circuitfn, vcircuitfn=vcircuitfn, masks=masks)
        # return super().compile(weights, lambda x: circuit(x, self.n, chi, m, vals))


class ColLogFuseEfn(FuseEfn):
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
        # weights = np.asarray([0, 1, 1])

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
        return "col"
