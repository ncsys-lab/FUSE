import time
from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import symengine as se
from jax.experimental import sparse


class Prob:
    @abstractmethod
    def gen_inst(self, key):
        pass

    def sol_inst(self, prob_inst):
        return 0


class Efn:
    pass


class ConvEfn(Efn):
    def __init__(self):
        print("[generate] Generating energy function...")
        (self.valid_expr, self.cost_expr, self.spins) = self._gen_exprs()

        square_dict = {spin**2: spin for spin in self.spins}
        self.cost_expr = self.cost_expr.expand().xreplace(square_dict)
        self.valid_expr = self.valid_expr.expand().xreplace(square_dict)

        zero_dict = {spin: 0 for spin in self.spins}
        self.zero_valid_expr = self.valid_expr.expand().xreplace(zero_dict)
        self.zero_cost_expr = self.cost_expr.expand().xreplace(zero_dict)

        print("[generate] Calculating symbolic gradients...")
        self.grad_valid_expr = [se.diff(self.valid_expr, spin) for spin in self.spins]
        self.grad_cost_expr = [se.diff(self.cost_expr, spin) for spin in self.spins]
        self.sparse = False

    def _gen_mats(self, n_spins, gradfn):
        h = jnp.array(gradfn(np.zeros(n_spins))).squeeze()
        J = jnp.array([gradfn(row).squeeze() - h for row in np.eye(n_spins)])
        return J, h

    def _lower_expr(
        self,
        in_expr,
        sub_dict,
        do_eval=False,
        sub_spins=None,
        in_grad_expr=None,
    ):
        expr = in_expr.xreplace(sub_dict).expand()
        if do_eval:
            return jnp.asarray(float(expr))

        assert (
            in_grad_expr is not None
        ), "Did not pass in gradient expression when do_eval was False!"

        if sub_spins is None:
            sub_spins = [s for s in self.spins if s in expr.free_symbols]

        n_spins = len(sub_spins)

        grad_expr = se.Matrix(in_grad_expr).xreplace(sub_dict)
        fgrad_expr = se.Matrix(
            [subexpr for subexpr, s in zip(grad_expr, self.spins) if s in sub_spins]
        )
        assert len(fgrad_expr) == n_spins

        return sub_spins, (fgrad_expr, expr)

    def compile(self, sub_dict):
        energy_syms, (valid_grad_expr, valid_ener_expr) = self._lower_expr(
            self.valid_expr, sub_dict, in_grad_expr=self.grad_valid_expr
        )
        n_spins = len(energy_syms)

        def compile_cse(in_expr):
            return se.LambdifyCSE([energy_syms], in_expr)

        _, (cost_grad_expr, cost_ener_expr) = self._lower_expr(
            self.cost_expr,
            sub_dict,
            sub_spins=energy_syms,
            in_grad_expr=self.grad_cost_expr,
        )

        if not self.sparse:
            bv = self._lower_expr(self.zero_valid_expr, sub_dict, do_eval=True)
            bc = self._lower_expr(self.zero_cost_expr, sub_dict, do_eval=True)

            valid_grad_fn = compile_cse(valid_grad_expr)
            cost_grad_fn = compile_cse(cost_grad_expr)

            Jv, hv = self._gen_mats(n_spins, valid_grad_fn)
            Jc, hc = self._gen_mats(n_spins, cost_grad_fn)

            J = Jc + Jv

            @jax.jit
            def engradfn(x, _):
                Jcx = jnp.dot(Jc, x)
                Jvx = jnp.dot(Jv, x)
                grad_c = Jcx + hc
                grad_v = Jvx + hv

                valid = jnp.dot(Jvx / 2 + hv, x) + bv
                cost = jnp.dot(Jcx / 2 + hc, x) + bc

                grad = grad_c + grad_v
                energy = valid + cost
                return (energy, (valid < 0.01), grad)

            dep_graph = nx.from_numpy_array(J)
            color_dict = nx.greedy_color(dep_graph)
            colors = np.fromiter(
                [color_dict[spin] for spin in range(n_spins)], dtype=int
            )

        else:
            grad_expr = valid_grad_expr + cost_grad_expr
            grad_fn = compile_cse(grad_expr)
            valid_fn = compile_cse(valid_ener_expr)
            cost_fn = compile_cse(cost_ener_expr)

            nx1_type = jax.ShapeDtypeStruct((n_spins, 1), dtype="float64")
            x1_type = jax.ShapeDtypeStruct((), dtype="float64")

            @jax.jit
            def engradfn(x, _):
                grad = jax.pure_callback(grad_fn, nx1_type, x)
                valid = jax.pure_callback(valid_fn, x1_type, x)
                cost = jax.pure_callback(cost_fn, x1_type, x)
                energy = valid + cost
                return (energy, (valid < 0.01), grad)

            dep_graph = nx.Graph()

            for sym in energy_syms:
                dep_graph.add_node(sym)

            for bit, expr in zip(energy_syms, grad_expr):
                for dep in expr.free_symbols:
                    dep_graph.add_edge(bit, dep)

            color_dict = nx.greedy_color(dep_graph)
            colors = np.fromiter([color_dict[sym] for sym in energy_syms], dtype=int)

        ncolors = max(colors) + 1
        print(f"[lower] Used p-bits: {len(energy_syms)}")
        print(f"[lower] Dependent colors: {ncolors}")

        masks = np.zeros((ncolors, len(energy_syms)), dtype=np.bool_)
        masks[colors, np.arange(colors.size)] = 1

        masks = jnp.asarray(masks)

        return engradfn, masks

    @abstractmethod
    def _gen_exprs(self):
        pass


class EncEfn(Efn):
    def __init__(self):
        print("[compile] Encocded Energy Function! Nothing to generate...")
        self.spins = 0

    def compile(self, weights, circuitfn, masks=None, vcircuitfn=None):
        if masks is None:

            @jax.jit
            def engradfn(state, mask):
                z_state = jnp.where(mask, 0, state)
                o_state = jnp.where(mask, 1, state)

                z_energy = jnp.dot(weights, circuitfn(z_state)).astype("float64")
                o_energy = jnp.dot(weights, circuitfn(o_state)).astype("float64")

                energy = jax.lax.select(jnp.dot(mask, state), o_energy, z_energy)
                return energy, True, o_energy - z_energy

            masks = jnp.asarray(np.eye(self.spins, dtype=np.bool_))

        else:

            @jax.jit
            def engradfn(state, mask):
                energy = jnp.dot(weights, circuitfn(state))
                flip_state = jnp.logical_xor(state, jnp.eye(self.spins))
                flip_energy = jnp.dot(vcircuitfn(flip_state), weights)
                grad = (2 * state - 1) * (energy - flip_energy)
                return energy, True, grad

        return engradfn, masks
