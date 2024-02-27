import jax
import jax.numpy as jnp
import json

from functools import partial

import jTWA

with open(__file__.rsplit("/", 1)[0] + "/test_config.json") as f:
    cfg = json.load(f)

cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)
samples = jTWA.spin1.initState.getPolarState(cfg)
operators = jTWA.spin1.observables.get_spin_operators(cfg)
hamiltonian = jTWA.spin1.hamiltonian.hamiltonian


def energy_expectation(samples, cfg):
    return jnp.mean(
        jax.vmap(hamiltonian, in_axes=(0, 0, None))(jnp.conj(samples), samples, cfg)
    )


def particle_number_expectation(samples):
    return jnp.mean(jnp.sum(jnp.abs(samples) ** 2, axis=(1, 2)))


def test_integrate():
    flow = jax.grad(partial(hamiltonian, cfg=cfg), argnums=0)
    samples_propagated = jTWA.integrate.integrate(samples, flow, 1e-3)

    assert (
        jnp.abs(
            energy_expectation(samples_propagated, cfg)
            - energy_expectation(samples, cfg)
        )
        < 1e-5
    )

    assert jnp.abs(
        particle_number_expectation(samples)
        - particle_number_expectation(samples_propagated)
        < 1e-10
    )
