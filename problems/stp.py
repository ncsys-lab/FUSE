import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from numpy.core.multiarray import where

from .prob import ConvEfn, EncEfn, Prob


class StpConvEfn(ConvEfn):
    def __init__(self, n, t, maxval):
        self.n = n
        self.t = t
        self.maxval = maxval
        super().__init__()
        self.sparse = True

    def _gen_funcs(self):
        N = self.n
        T = self.t
        max_vd = N // 2 + 1
        max_ed = N // 2
        combos = N * (N - 1) // 2

        idx = jnp.triu_indices(N, k=1)
        off_diag = ~jnp.eye(N, dtype=bool)

        @jax.jit
        def dispatch_fn(spins):
            dispatch_idx = [
                N * max_vd,
                N * max_vd + N - T,
                N * max_vd + N - T + combos,
            ]
            return jnp.split(spins, dispatch_idx)

        def valid_fn(spins, inst):
            v_depth, v_sel, e_sel, pre_depth = dispatch_fn(spins)

            adjust_derivs = (
                (2 * v_depth * (v_depth - 1)).sum()
                + (v_sel * (v_sel - 1)).sum()
                + (e_sel * (e_sel - 1)).sum()
                + (2 * pre_depth * (pre_depth - 1)).sum()
            )

            v_depth = v_depth.reshape(N, max_vd)
            e_depth = jnp.zeros((N, N, max_ed))
            e_depth = e_depth.at[off_diag].set(pre_depth.reshape((-1, max_ed)))

            one_root = (1 - v_depth[:, 0].sum()) ** 2
            cond_depth = jnp.hstack((jnp.ones(T, dtype=int), v_sel))
            one_depth = ((cond_depth - v_depth.sum(axis=-1)) ** 2).sum()

            one_depth_edge = (
                (e_sel - (e_depth[idx] + e_depth.swapaxes(0, 1)[idx]).sum(axis=-1)) ** 2
            ).sum()

            masked_depth = e_depth.swapaxes(0, 1)[off_diag].reshape(N, N - 1, max_ed)
            edge_lower = ((v_depth[:, 1:] - masked_depth.sum(axis=1)) ** 2).sum()

            edge_depth_adj = (
                e_depth * (2 - v_depth[:, :-1][:, jnp.newaxis] - v_depth[:, 1:])
            )[off_diag].sum()
            return (self.n * self.maxval) * (
                one_root
                + one_depth
                + one_depth_edge
                + edge_lower
                + edge_depth_adj
                - adjust_derivs
            )

        def cost_fn(spins, inst):
            _, _, e_sel, _ = dispatch_fn(spins)
            return jnp.dot(e_sel, inst)

        n_spins = N * max_vd + N - T + combos + N * (N - 1) * max_ed
        return valid_fn, cost_fn, n_spins


class StpEncEfn(EncEfn):
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.spins = self.n - self.t

    def compile(self, inst):
        adj_mat = jnp.zeros(shape=(self.n, self.n))
        adj_mat = adj_mat.at[jnp.triu_indices_from(adj_mat, k=1)].set(inst)
        adj_mat += adj_mat.T

        @jax.jit
        def circuitfn(state):
            n = self.t + state.sum()
            parents = jnp.full(self.n, -1, dtype=int)
            incs = jnp.append(jnp.zeros(self.t, dtype=bool), ~state)

            keys = jnp.full(self.n, jnp.inf)
            keys = keys.at[0].set(0)

            def inner_loop(state):
                i, inc, key, parent = state
                min_v = jnp.argmin(jnp.where(inc, jnp.inf, key))
                inc = inc.at[min_v].set(True)
                cond = ~inc & (adj_mat[min_v] < key)
                key = jnp.where(cond, adj_mat[min_v], key)
                parent = jnp.where(cond, min_v, parent)
                return (i + 1, inc, key, parent)

            def cond_fn(state):
                i, _, _, _ = state
                return i < n - 1

            state = (0, incs, keys, parents)
            _, _, _, parent = jax.lax.while_loop(cond_fn, inner_loop, state)

            res = jnp.zeros((self.n, self.n), dtype=bool)
            idx_0 = jnp.where(parent != -1, jnp.arange(self.n), -1)
            res = res.at[idx_0, parent].set(True)
            res += res.T
            return res[jnp.triu_indices(self.n, k=1)]

        return super().compile(inst, circuitfn)


class Stp(Prob):
    def __init__(self, args):
        self.n = args.size
        self.t = args.terminals
        self.minval = args.minval
        self.maxval = args.maxval
        if args.enc:
            self.efn = StpEncEfn(self.n, self.t)
        else:
            self.efn = StpConvEfn(self.n, self.t, self.maxval)

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

        term_nodes = np.arange(self.t)
        best_tree = nx.approximation.steiner_tree(g, term_nodes, method="kou")
        best_weight = best_tree.size(weight="weight")
        return best_weight

    @staticmethod
    def gen_parser(subparser):
        parser = subparser.add_parser("stp", help="Steiner Tree Problem")
        parser.add_argument("-n", "--size", type=int, default=8)
        parser.add_argument("-u", "--terminals", type=int, default=4)
        parser.add_argument("-minval", type=int, default=1)
        parser.add_argument("-maxval", type=int, default=100)
        return "stp"
