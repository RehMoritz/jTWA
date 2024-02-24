import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def getPolarState(cfg):
    key = jax.random.PRNGKey(cfg["simulationParameters"]["random_seed"])
    samples = jax.random.normal(
        key,
        shape=(
            int(cfg["simulationParameters"]["n_samples"]),
            int(cfg["systemParameters"]["n_wells"]),
            3,
            2,
        ),
    )

    samples = (
        (samples[..., 0] + 1j * samples[..., 1])
        / 2
        * jnp.sqrt(1 + 2 / (jnp.exp(cfg["systemParameters"]["beta"]) - 1))
    )

    samples = samples.at[:, :, 1].set(
        samples[:, :, 1] + jnp.sqrt(cfg["systemParameters"]["n_atoms_per_well"])
    )

    return samples
