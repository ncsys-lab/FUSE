from abc import abstractmethod

import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy


class Prob:
    @abstractmethod
    def gen_inst(self, key):
        pass

    @abstractmethod
    def sol_inst(self, prob_inst):
        pass


class Efn:
    pass


class ConvEfn(Efn):
    def __init__(self):
        print("[compile] Generating energy function...")
        (self.energy_expr, self.spins) = self._gen_exprs()

        print("[compile] Calculating symbolic gradients...")
        self.grad_expr = sympy.Matrix(
            [sympy.diff(self.energy_expr, spin) for spin in self.spins]
        )

    def compile(self, sub_dict):
        energy_expr = self.energy_expr.xreplace(sub_dict)
        grad_expr = self.grad_expr.xreplace(sub_dict)

        energyfn = sympy.lambdify([self.spins], energy_expr, "jax")
        gradfn = sympy.lambdify([self.spins], grad_expr, "jax")

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

        return energyfn, gradfn, masks

    def sol_inst(self, prob_inst):
        return 0

    @abstractmethod
    def _gen_exprs(self):
        pass


class FuseEfn(Efn):
    pass
