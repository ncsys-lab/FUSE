import time
from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import symengine as se
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

        square_dict = {spin**2: spin for spin in self.spins}
        self.energy_expr = self.energy_expr.expand().xreplace(square_dict)

        print("[generate] Calculating symbolic gradients...")
        self.grad_expr = se.Matrix(
            [se.diff(self.energy_expr, spin) for spin in self.spins]
        )

    def compile(self, sub_dict):
        energy_expr = self.energy_expr.xreplace(sub_dict).expand()

        grad_expr = self.grad_expr.xreplace(sub_dict)

        energy_syms = [s for s in self.spins if s in energy_expr.free_symbols]
        grad_syms = [s for s in self.spins if s in grad_expr.free_symbols]

        fgrad_expr = se.Matrix(
            [
                expr
                for (expr, s) in zip(grad_expr, self.spins)
                if s in energy_expr.free_symbols
            ]
        )

        zero_dict = {spin: 0 for spin in energy_syms}
        bias = jnp.array(float(energy_expr.subs(zero_dict)))

        segrad = se.LambdifyCSE([energy_syms], fgrad_expr)

        n_spins = len(energy_syms)
        h = jnp.array(segrad(np.zeros(n_spins))).squeeze()
        J = jnp.array([segrad(row).squeeze() - h for row in np.eye(n_spins)])

        @jax.jit
        def engradfn(x, _):
            grad = jnp.dot(J, x) + h
            energy = jnp.dot(grad, x) + bias
            return grad, energy

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

        return engradfn, masks

    @abstractmethod
    def _gen_exprs(self):
        pass


class FuseEfn(Efn):
    def __init__(self):
        print("[compile] Fuse Function! Nothing to generate...")
        self.spins = 0

    def compile(self, weights, circuitfn, masks=None):
        @jax.jit
        def engradfn(state, mask):
            z_state = jnp.where(mask, 0, state)
            o_state = jnp.where(mask, 1, state)

            z_energy = jnp.dot(weights, circuitfn(z_state))
            o_energy = jnp.dot(weights, circuitfn(o_state))

            energy = jax.lax.select(jnp.dot(mask, state), o_energy, z_energy)
            return energy, o_energy - z_energy

        if masks is None:
            masks = jnp.asarray(np.eye(self.spins, dtype=np.bool_))
        return engradfn, masks
