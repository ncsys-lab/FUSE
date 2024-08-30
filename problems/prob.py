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

        gradfn = sympy.lambdify([self.spins], grad_expr, modules="jax", cse=True)
        energyfn = sympy.lambdify([self.spins], energy_expr, modules="jax", cse=True)

        g = nx.Graph()
        for spin in self.spins:
            g.add_node(spin)

        for spin, expr in zip(self.spins, grad_expr):
            for dep in expr.free_symbols:
                g.add_edge(spin, dep)

        color_dict = nx.greedy_color(g)
        colors = np.fromiter([color_dict[spin] for spin in self.spins], dtype=int)
        ncolors = max(colors) + 1

        masks = np.zeros((ncolors, len(self.spins)), dtype=np.bool_)
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
