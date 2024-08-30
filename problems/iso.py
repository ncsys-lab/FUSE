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
        combos = self.n * (self.n - 1) // 2
        spins = np.array(sympy.symbols(f"s:{self.n*self.n}")).reshape((self.n, self.n))
        g1_edges = np.array(sympy.symbols(f"a:{combos}"))
        g2_edges = np.array(sympy.symbols(f"b:{combos}"))

        idx = np.array(np.triu_indices_from(spins, k=1))

        # if edge_nedge pair is AB_CD, then
        ac = spins[np.meshgrid(idx[0], idx[0], indexing="ij")]
        bd = spins[np.meshgrid(idx[1], idx[1], indexing="ij")]
        ad = spins[np.meshgrid(idx[0], idx[1], indexing="ij")]
        bc = spins[np.meshgrid(idx[1], idx[0], indexing="ij")]

        edge_matrix = ac * bd + ad * bc

        g1_nedges = 1 - g1_edges
        g2_nedges = 1 - g2_edges

        cost_expr = sympy.Add(
            *(
                (
                    g1_edges * g2_nedges[:, np.newaxis]
                    + g2_edges * g1_nedges[:, np.newaxis]
                )
                * edge_matrix
            ).flatten()
        )

        g1_once = ((1 - spins.sum(axis=0)) ** 2).sum()
        g2_once = ((1 - spins.sum(axis=1)) ** 2).sum()
        invalid_expr = (self.n) * (sympy.Add(g1_once, g2_once))

        energy_expr = invalid_expr + cost_expr

        self.g1 = g1_edges
        self.g2 = g2_edges
        return energy_expr, spins.flatten()

    def compile(self, inst):
        g1, g2 = inst
        g1_dict = {edge_v: edge for edge_v, edge in zip(self.g1, g1)}
        g2_dict = {edge_v: edge for edge_v, edge in zip(self.g2, g2)}
        sub_dict = {**g1_dict, **g2_dict}
        return super().compile(sub_dict)


class IsoFuseEfn(FuseEfn):
    def __init__(self, n):
        self.n = n
        self.spins, permutefn = PermuteNet(self.n)

        idx = jnp.triu_indices(self.n, k=1)
        ac_idx = tuple(jnp.meshgrid(idx[0], idx[0], indexing="ij"))
        bd_idx = tuple(jnp.meshgrid(idx[1], idx[1], indexing="ij"))
        ad_idx = tuple(jnp.meshgrid(idx[0], idx[1], indexing="ij"))
        bc_idx = tuple(jnp.meshgrid(idx[1], idx[0], indexing="ij"))

        @jax.jit
        def circuitfn(state):
            spins = permutefn(state)
            ac = spins[ac_idx]
            bd = spins[bd_idx]
            ad = spins[ad_idx]
            bc = spins[bc_idx]

            return (ac * bd + ad * bc).flatten()

        self.circuitfn = circuitfn

    def compile(self, inst):
        g1_edges, g2_edges = inst
        g1_nedges = 1 - g1_edges
        g2_nedges = 1 - g2_edges
        weights = (
            g1_edges * g2_nedges[:, np.newaxis] + g2_edges * g1_nedges[:, np.newaxis]
        ).flatten()
        return super().compile(weights, self.circuitfn)


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
        key, *keys = jax.random.split(key, num=3)

        edges = np.asarray(
            jax.random.bernoulli(keys[0], p=self.nu, shape=combos)
        ).astype(int)

        g1_mat = np.zeros((self.n, self.n), dtype=int)
        g1_mat[np.triu_indices(self.n, k=1)] = edges
        g1_mat += g1_mat.T

        perm_mat = jax.random.permutation(keys[1], jnp.eye(self.n, dtype=int))
        g2_mat = perm_mat @ g1_mat @ perm_mat.T
        edges_perm = g2_mat[np.triu_indices(self.n, k=1)]

        return edges, edges_perm

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("iso", help="Graph Isomorphism Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-nu", "--connectivity", type=float, default=0.5)
        return "iso"
