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
