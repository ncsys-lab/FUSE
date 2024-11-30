import math

import jax
import jax.numpy as jnp


class PermuteNet:
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
