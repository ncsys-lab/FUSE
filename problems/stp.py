import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import symengine as se

from circuit import PermuteNet

from .prob import ConvEfn, EncEfn, Prob


class StpConvEfn(ConvEfn):
    def __init__(self, n, t, maxval):
        self.n = n
        self.t = t
        self.maxval = maxval
        super().__init__()

    def _gen_exprs(self):
        N = self.n
        T = self.t
        max_vd = N // 2 + 1
        max_ed = N // 2
        combos = N * (N - 1) // 2

        v_depth = np.array(se.symbols(f"vd:{N * max_vd}")).reshape(N, max_vd)
        v_sel = np.array(se.symbols(f"v:{N - T}"))
        e_sel = np.array(se.symbols(f"e:{combos}"))
        e_depth = np.array(se.symbols(f"ed:{N * N * max_ed}")).reshape(N, N, max_ed)
        weights = np.array(se.symbols(f"w:{self.n*(self.n-1)//2}"))

        idx = np.triu_indices(N, k=1)
        off_diag = ~np.eye(N, dtype=bool)

        spins = np.hstack(
            (v_depth.flatten(), v_sel, e_sel, e_depth[off_diag].flatten())
        )

        cost_expr = np.dot(e_sel, weights)

        one_root = (1 - v_depth[:, 0].sum()) ** 2
        cond_depth = np.hstack((np.ones(T, dtype=int), v_sel))
        one_depth = ((cond_depth - v_depth.sum(axis=-1)) ** 2).sum()

        one_depth_edge = (
            (e_sel - (e_depth[idx] + e_depth.swapaxes(0, 1)[idx]).sum(axis=-1)) ** 2
        ).sum()

        masked_depth = e_depth.swapaxes(0, 1)[off_diag].reshape(N, N - 1, max_ed)
        edge_lower = ((v_depth[:, 1:] - masked_depth.sum(axis=1)) ** 2).sum()

        edge_depth_adj = (
            e_depth * (2 - v_depth[:, :-1][:, np.newaxis] - v_depth[:, 1:])
        )[off_diag].sum()

        invalid_expr = (N * self.maxval) * (
            se.Add(one_root, one_depth, one_depth_edge, edge_lower, edge_depth_adj)
        )

        energy_expr = invalid_expr + cost_expr

        self.weights = weights
        return energy_expr, spins.flatten()

    def compile(self, inst):
        sub_dict = {weight: inst_w for weight, inst_w in zip(self.weights, inst)}
        return super().compile(sub_dict)


class StpEncEfn(EncEfn):
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.spins = self.n - self.t

    def compile(self, inst):
        g = nx.Graph()
        for i in range(self.n):
            g.add_node(i)

        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                g.add_edge(i, j, weight=inst[idx])
                idx += 1

        term_nodes = np.arange(self.t)
        st_nodes = np.arange(self.t, self.n)

        def mst_callback(state):
            ind_nodes = np.append(term_nodes, st_nodes[state.astype(bool)])
            ind_g = nx.induced_subgraph(g, ind_nodes)
            edges = nx.minimum_spanning_tree(ind_g, algorithm="prim").edges
            view = nx.subgraph_view(g, filter_edge=lambda u, v: (u, v) in edges)
            out = nx.to_numpy_array(view, weight=None)[
                np.triu_indices(self.n, k=1)
            ].astype(np.int16)
            return out

        @jax.jit
        def circuitfn(state):
            return jax.pure_callback(
                mst_callback,
                jax.ShapeDtypeStruct((self.n * (self.n - 1) // 2,), dtype="int16"),
                state,
            )

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
