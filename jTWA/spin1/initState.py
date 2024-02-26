import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def getPolarState(cfg):
    """
    Returns samples of the Wigner distribution representing the polar state, with all atoms in the zero mode, so that

    .. math ::

        |\\psi\\rangle = |0, \\alpha, 0\\rangle^{\\otimes N_{wells}}.

    Note that it is also possible to set a temperature :math:`\\beta` that broadens the distribution, accounting for thermal fluctuations.

    Args:
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * ``samples``: An array of samples of shape (:math:`N_{samples}`, :math:`N_{wells}`, :math:`N_{internal}`).
    """
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
