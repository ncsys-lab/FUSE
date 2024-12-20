import functools as ft
from abc import abstractmethod

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from amaranth.back import verilog
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from scipy.sparse import csr_matrix

epsilon = 0.0001


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
        print("[generate] Generating energy functions...")
        self.valid_fn, self.cost_fn, self.spins = self._gen_funcs()
        self.sparse = False

    def gen_circuit(self, dir):
        with open(f"{dir}/src/params.v", "w+") as f:
            f.write(f"localparam PBITS = {self.pbits};\n")

    def compile(self, inst):
        valid_fn = jax.jit(ft.partial(self.valid_fn, inst=inst))
        cost_fn = jax.jit(ft.partial(self.cost_fn, inst=inst))

        Jv = jax.hessian(valid_fn)(jnp.zeros(shape=(self.spins,)))
        Jc = jax.hessian(cost_fn)(jnp.zeros(shape=(self.spins,)))
        J = Jv + Jc

        assert jnp.all(
            jnp.abs(jnp.diagonal(J)) < epsilon
        ), "Diagonal Entries of Hessian must be 0!"

        valid_vgrad_fn = jax.value_and_grad(valid_fn)
        cost_vgrad_fn = jax.value_and_grad(cost_fn)

        @jax.jit
        def engradfn(x, _):
            valid, grad_v = valid_vgrad_fn(x)
            cost, grad_c = cost_vgrad_fn(x)

            energy = valid + cost
            grad = grad_c + grad_v
            return (energy, (valid < epsilon), grad)

        print(f"[lower] Used p-bits: {self.spins}")
        J_mat = csr_matrix(J) if self.sparse else J
        dep_graph = nx.from_numpy_array(J_mat)
        color_dict = nx.greedy_color(dep_graph)
        colors = np.fromiter(
            [color_dict[spin] for spin in range(self.spins)], dtype=int
        )

        ncolors = max(colors) + 1
        print(f"[lower] Dependent colors: {ncolors}")

        masks = np.zeros((ncolors, self.spins), dtype=np.bool_)
        masks[colors, np.arange(colors.size)] = 1

        masks = jnp.asarray(masks)

        return engradfn, masks

    @abstractmethod
    def _gen_funcs(self):
        pass


class EncEfn(Efn, wiring.Component):
    def __init__(self, wire_dict={}):
        print("[generate] Encocded Energy Function! Nothing to generate...")
        wiring.Component.__init__(self, wire_dict)

        if wire_dict:
            self.pbit_shape = wire_dict["state"].shape
            self.output_shape = wire_dict["outputs"].shape

    def gen_circuit(self, dir):
        with open(f"{dir}/src/enc_circuit.v", "w+") as f:
            f.write(verilog.convert(self, name="enc_circuit"))

        with open(f"{dir}/src/params.v", "w+") as f:
            f.write(f"localparam PBITS = {self.pbit_shape};\n")
            f.write(f"localparam OUTPUTS = {self.output_shape};\n")

    def compile(self, weights, circuitfn, masks=None, vcircuitfn=None):
        if masks is None:

            @jax.jit
            def engradfn(state, mask):
                z_state = jnp.where(mask, False, state)
                o_state = jnp.where(mask, True, state)

                z_energy = jnp.dot(weights, circuitfn(z_state)).astype("float64")
                o_energy = jnp.dot(weights, circuitfn(o_state)).astype("float64")

                energy = jax.lax.select(jnp.dot(mask, state), o_energy, z_energy)
                return energy, True, o_energy - z_energy

            masks = jnp.asarray(np.eye(self.spins, dtype=bool))

        else:

            @jax.jit
            def engradfn(state, mask):
                energy = jnp.dot(weights, circuitfn(state))
                flip_state = jnp.logical_xor(state, jnp.eye(self.spins))
                flip_energy = jnp.dot(vcircuitfn(flip_state), weights)
                grad = (2 * state - 1) * (energy - flip_energy)
                return energy, True, grad

        return engradfn, masks
