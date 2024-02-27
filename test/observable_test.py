import jax
import jax.numpy as jnp
import json

import jTWA

with open("test_config.json") as f:
    cfg = json.load(f)

cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)
samples = jTWA.spin1.initState.getPolarState(cfg)
operators = jTWA.spin1.observables.get_spin_operators(cfg)


def test_get_spin_operators():
    assert operators["operators"].shape[0] == len(operators["names"])
    assert operators["operators"].shape == (
        len(cfg["simulationParameters"]["obs"]),
        3,
        3,
    )
    for o in operators["operators"]:
        assert jnp.all(jnp.conj(o).T == o)


def test_compute_observables():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, samples.shape[0])

    res = jax.vmap(
        jTWA.spin1.observables.compute_observables, in_axes=(0, 0, None, None)
    )(
        samples,
        keys,
        operators["operators"],
        jnp.sqrt(2 * cfg["systemParameters"]["n_atoms_per_well"]),
    )

    assert res["atom_number_realspace"].shape == samples.shape
    assert res["atom_number_momspace"].shape == samples.shape
    assert res["spin_obs"].shape == samples.shape[:2] + (len(operators["names"]),)
    assert res["spin_obs_sim"].shape == samples.shape[:2] + (len(operators["names"]),)

    assert jnp.abs(jnp.sum(jnp.imag(res["atom_number_realspace"]))) < 1e-12
    assert jnp.abs(jnp.sum(jnp.imag(res["atom_number_momspace"]))) < 1e-12
    assert jnp.abs(jnp.sum(jnp.imag(res["spin_obs"]))) < 1e-12
    assert jnp.abs(jnp.sum(jnp.imag(res["spin_obs_sim"]))) < 1e-12

    assert jnp.all(jnp.abs(jnp.mean(res["spin_obs"], axis=0)) < 1e-1)
    assert jnp.all(
        jnp.abs(
            jnp.diag(jnp.cov(res["spin_obs"].reshape(samples.shape[0], -1).T)) - 0.5
        )
        < 5e-2
    )
    assert jnp.all(
        jnp.abs(jnp.std(res["spin_obs"], axis=0))
        < jnp.abs(jnp.std(res["spin_obs_sim"], axis=0))
    )

    assert jnp.all(
        jnp.diag(jnp.cov(res["spin_obs"].reshape(samples.shape[0], -1).T))
        < jnp.diag(jnp.cov(res["spin_obs_sim"].reshape(samples.shape[0], -1).T))
    )
