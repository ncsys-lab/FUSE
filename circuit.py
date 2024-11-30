import math

import amaranth as am
import jax
import jax.numpy as jnp
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class PermuteNet(wiring.Component):
    def __init__(self, n):
        self.n = n
        self.swaps = []

        def make_permute(inputs):
            n = len(inputs)
            if n <= 1:
                return
            for i in range(0, n - 1, 2):
                self.swaps.append((inputs[i], inputs[i + 1]))
            if n == 2:
                return
            top_inputs = [el for i, el in enumerate(inputs) if i % 2 == 0]
            bot_inputs = [el for i, el in enumerate(inputs) if i % 2 == 1]
            make_permute(top_inputs)
            make_permute(bot_inputs)
            for i in range(0, n - 2, 2):
                self.swaps.append((inputs[i], inputs[i + 1]))

        make_permute(range(n))

        self.n_spins = len(self.swaps)
        self.jcomps = jnp.array([(y, x) for x, y in self.swaps], dtype=int)
        self.jswaps = jnp.array(self.swaps, dtype=int)

        # Verilog
        super().__init__(
            {
                "state": In(self.n_spins),
                "outputs": Out(self.n * self.n),
            }
        )

    def circuitfn(self):
        @jax.jit
        def out_fn(spins):
            def swap(i, state):
                return state.at[self.jswaps[i]].set(
                    jax.lax.select(
                        spins[i], state[self.jcomps[i]], state[self.jswaps[i]]
                    )
                )

            state = jnp.eye(self.n)
            return jax.lax.fori_loop(0, self.n_spins, swap, state)

        return out_fn

    def elaborate(self, _) -> am.Module:
        m = am.Module()

        frontier = [am.Const(1 << i) for i in range(self.n)]

        for i, swap in enumerate(self.swaps):
            j = swap[0]
            k = swap[1]
            sig_0 = am.Signal(self.n)
            sig_1 = am.Signal(self.n)
            m.d.comb += sig_0.eq(am.Mux(self.state[i], frontier[k], frontier[j]))
            m.d.comb += sig_1.eq(am.Mux(self.state[i], frontier[j], frontier[k]))
            frontier[j] = sig_0
            frontier[k] = sig_1

        for i, signal in enumerate(frontier):
            m.d.comb += self.outputs[i * self.n : (i + 1) * self.n].eq(signal)
        return m


# N replicas of selecting b/w K items
class SelectNet:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.swaps = []

        def make_select(index, len):
            if len == 1:
                return

            len_round = 1 << math.floor(math.log(len, 2))

            half_len = len_round // 2
            make_select(index, half_len)
            make_select(index + half_len, half_len)
            self.swaps.append((index, index + half_len))

            if len != len_round:
                make_select(index + len_round, len - len_round)
                self.swaps.append((index, index + len_round))

        make_select(0, k)

        self.n_spins = n * (k - 1)
        self.jcomps = jnp.array([(y, x) for x, y in self.swaps], dtype=int)
        self.jswaps = jnp.array(self.swaps, dtype=int)

    def circuitfn(self):
        @jax.jit
        def out_fn(spins):
            def swap(i, state):
                return state.at[self.jswaps[i]].set(
                    jax.lax.select(
                        spins[i], state[self.jcomps[i]], state[self.jswaps[i]]
                    )
                )

            state = jnp.eye(self.n)
            out = jax.lax.fori_loop(0, self.k - 1, swap, state)
            return out[0]

        vout_fn = jax.vmap(out_fn)
        return vout_fn
