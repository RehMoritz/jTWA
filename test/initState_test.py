import jTWA
import jax.numpy as jnp
import json


def test_initState():
    with open(__file__.rsplit("/", 1)[0] + "/test_config.json") as f:
        cfg = json.load(f)

    samples = jTWA.spin1.initState.getPolarState(cfg)

    noise(samples)
    occupations(samples, cfg)


def noise(samples):
    assert jnp.all(jnp.abs(jnp.std(samples, axis=0) - 1 / jnp.sqrt(2)) < 4e-2)


def occupations(samples, cfg):
    assert jnp.all(
        jnp.abs(
            jnp.mean(jnp.sum(jnp.abs(samples) ** 2, axis=(2,)), axis=(0,))
            - cfg["systemParameters"]["n_atoms_per_well"]
        )
        < 10
    )
