import jax
import jax.numpy as jnp
import json

import jTWA


def test_hamiltonian():
    with open(__file__.rsplit("/", 1)[0] + "/test_config.json") as f:
        cfg = json.load(f)

    cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)
    samples = jTWA.spin1.initState.getPolarState(cfg)

    is_hermitian(samples, cfg)


def is_hermitian(samples, cfg):
    a = jax.vmap(jTWA.spin1.hamiltonian.hamiltonian, in_axes=(0, 0, None))(
        jnp.conj(samples), samples, cfg
    )
    b = jax.vmap(jTWA.spin1.hamiltonian.hamiltonian, in_axes=(0, 0, None))(
        samples, jnp.conj(samples), cfg
    )

    assert jnp.allclose(a, b)
