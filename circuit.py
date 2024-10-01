import math

import jax
import jax.numpy as jnp


def PermuteNet(n):
    swaps = []

    def make_permute(inputs):
        n = len(inputs)
        if n <= 1:
            return
        for i in range(0, n - 1, 2):
            swaps.append((inputs[i], inputs[i + 1]))
        if n == 2:
            return
        top_inputs = [el for i, el in enumerate(inputs) if i % 2 == 0]
        bot_inputs = [el for i, el in enumerate(inputs) if i % 2 == 1]
        make_permute(top_inputs)
        make_permute(bot_inputs)
        for i in range(0, n - 2, 2):
            swaps.append((inputs[i], inputs[i + 1]))

    make_permute(range(n))

    n_spins = len(swaps)
    comps = jnp.array([(y, x) for x, y in swaps], dtype=int)
    swaps = jnp.array(swaps, dtype=int)

    @jax.jit
    def circuitfn(spins):
        def swap(i, state):
            return state.at[swaps[i]].set(
                jax.lax.select(spins[i], state[comps[i]], state[swaps[i]])
            )

        state = jnp.eye(n)
        return jax.lax.fori_loop(0, n_spins, swap, state)

    return n_spins, circuitfn


# N replicas of selecting b/w K items
def SelectNet(n, k):
    swaps = []

    def make_select(index, len):
        if len == 1:
            return

        len_round = 1 << math.floor(math.log(len, 2))

        half_len = len_round // 2
        make_select(index, half_len)
        make_select(index + half_len, half_len)
        swaps.append((index, index + half_len))

        if len != len_round:
            make_select(index + len_round, len - len_round)
            swaps.append((index, index + len_round))
        print(swaps)

    make_select(0, k)

    n_spins = n * (k - 1)
    comps = jnp.array([(y, x) for x, y in swaps], dtype=int)
    swaps = jnp.array(swaps, dtype=int)

    @jax.jit
    def circuitfn(spins):
        def swap(i, state):
            return state.at[swaps[i]].set(
                jax.lax.select(spins[i], state[comps[i]], state[swaps[i]])
            )

        state = jnp.eye(n)
        out = jax.lax.fori_loop(0, k - 1, swap, state)
        return out[0]

    vcircuitfn = jax.vmap(circuitfn)
    return n_spins, vcircuitfn

    return n * k, circuitfn
