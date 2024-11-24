import time
from abc import abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import symengine as se


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

        zero_dict = {spin: 0 for spin in self.spins}
        self.zero_expr = self.energy_expr.expand().xreplace(zero_dict)

        print("[generate] Calculating symbolic gradients...")
        self.grad_expr = [se.diff(self.energy_expr, spin) for spin in self.spins]

    def compile(self, sub_dict):
        energy_expr = self.energy_expr.xreplace(sub_dict).expand()
        zero_expr = self.zero_expr.xreplace(sub_dict)
        grad_expr = se.Matrix(self.grad_expr).xreplace(sub_dict)

        energy_syms = [s for s in self.spins if s in energy_expr.free_symbols]

        n_spins = len(energy_syms)

        bias = jnp.asarray(float(zero_expr))

        fgrad_expr = se.Matrix(
            [
                expr
                for (expr, s) in zip(grad_expr, self.spins)
                if s in energy_expr.free_symbols
            ]
        )
        segrad = se.LambdifyCSE([energy_syms], fgrad_expr)
        # seenergy = se.LambdifyCSE([energy_syms], energy_expr)

        h = jnp.array(segrad(np.zeros(n_spins))).squeeze()
        J = jnp.array([segrad(row).squeeze() - h for row in np.eye(n_spins)]) / 2

        @jax.jit
        def engradfn(x, _):
            Jx = jnp.dot(J, x)
            grad = Jx + h
            energy = jnp.dot(grad, x) + bias

            """
            grad_1 = jax.pure_callback(
                segrad,
                jax.ShapeDtypeStruct(
                    (n_spins, 1),
                    dtype="float64",
                ),
                x,
            ).reshape((n_spins,))

            jax.debug.print(
                "state:{state}\ngrad:\t{grad}\t{grad_1}",
                state=x,
                grad=grad,
                grad_1=grad_1,
            )

            energy_1 = jax.pure_callback(
                seenergy,
                jax.ShapeDtypeStruct(
                    (),
                    dtype="float64",
                ),
                x,
            )
            jax.debug.print(
                "ener_ALLCLOSE:\t{close}",
                close=jnp.allclose(energy, energy_1),
            )
            """

            return (energy, grad)

        color_dict = nx.greedy_color(nx.from_numpy_array(J))
        colors = np.fromiter([color_dict[spin] for spin in range(n_spins)], dtype=int)

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
                return energy, o_energy - z_energy

            masks = jnp.asarray(np.eye(self.spins, dtype=np.bool_))

        else:

            @jax.jit
            def engradfn(state, mask):
                energy = jnp.dot(weights, circuitfn(state))
                flip_state = jnp.logical_xor(state, jnp.eye(self.spins))
                flip_energy = jnp.dot(vcircuitfn(flip_state), weights)
                grad = (2 * state - 1) * (energy - flip_energy)
                return energy, grad

        return engradfn, masks
