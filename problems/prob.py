from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy


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
        (self.energy_expr, self.spins) = self._gen_exprs()

        print("[generate] Calculating symbolic gradients...")
        self.grad_expr = sympy.Matrix(
            [sympy.diff(self.energy_expr, spin) for spin in self.spins]
        )

    def compile(self, sub_dict):
        energy_expr = self.energy_expr.xreplace(sub_dict)
        grad_expr = self.grad_expr.xreplace(sub_dict)

        fgrad_expr = sympy.Matrix(
            [
                expr
                for (expr, s) in zip(grad_expr, self.spins)
                if s in energy_expr.free_symbols
            ]
        )

        energy_syms = [s for s in self.spins if s in energy_expr.free_symbols]
        grad_syms = [s for s in self.spins if s in grad_expr.free_symbols]

        assert energy_syms == grad_syms

        gradfn = sympy.lambdify([energy_syms], fgrad_expr, modules="jax", cse=True)
        energyfn = sympy.lambdify([grad_syms], energy_expr, modules="jax", cse=True)

        g = nx.Graph()
        for spin in energy_syms:
            g.add_node(spin)

        for spin, expr in zip(energy_syms, fgrad_expr):
            for dep in expr.free_symbols:
                g.add_edge(spin, dep)

        color_dict = nx.greedy_color(g)
        colors = np.fromiter([color_dict[spin] for spin in energy_syms], dtype=int)
        ncolors = max(colors) + 1
        masks = np.zeros((ncolors, len(energy_syms)), dtype=np.bool_)
        masks[colors, np.arange(colors.size)] = 1
        masks = jnp.asarray(masks)

        return energyfn, (lambda x, _: gradfn(x)), masks

    @abstractmethod
    def _gen_exprs(self):
        pass


class FuseEfn(Efn):
    def __init__(self):
        print("[compile] Fuse Function! Nothing to generate...")
        self.spins = 0

    def compile(self, weights, circuitfn, masks=None):
        def gradfn(state, mask):
            z_state = jnp.where(mask, 0, state)
            o_state = jnp.where(mask, 1, state)
            return jnp.dot(weights, circuitfn(o_state) - circuitfn(z_state))

        def energyfn(state):
            return jnp.dot(weights, circuitfn(state))

        if masks is None:
            masks = jnp.asarray(np.eye(self.spins, dtype=np.bool_))
        return energyfn, gradfn, masks
