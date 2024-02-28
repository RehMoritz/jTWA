import jax
import jax.numpy as jnp


def update_cfg(cfg):
    """
    Since the energy scale is set by :math:`\\langle N \\rangle c_1 = -1` we have :math:`c_1=-1/\\langle N \\rangle`.
    This computation is done at runtime to minimize the risk of inconsistencies, when setting :math:`c_1` and :math:`\\langle N \\rangle` separately.
    However, this also means that all quantities that depend on :math:`c_1` must be computed at runtime as well.
    This is done within this function.

    Args:
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * ``cfg``: The altered dictionary that contains the settings of the current run.
    """

    cfg["hamiltonianParameters"]["c_1"] = (
        -1 / cfg["systemParameters"]["n_atoms_per_well"]
    )
    cfg["hamiltonianParameters"]["c_0"] = (
        cfg["hamiltonianParameters"]["c_0/c_1"] * cfg["hamiltonianParameters"]["c_1"]
    )
    cfg["hamiltonianParameters"]["p"] = (
        cfg["hamiltonianParameters"]["p/c_1"] * cfg["hamiltonianParameters"]["c_1"]
    )

    return cfg


def hamiltonian(sample_conj, sample, cfg):
    """
    Defines the hamiltonian of the full system.
    The function takes both the sample and its complex conjugate as input, so that it is easy to obtain its time derivative

    .. math ::

        i\\partial_t \\alpha = \\partial_{\\alpha^*} H(\\alpha, \\alpha^*).

    Here :math:`\\alpha` is the vector of length :math:`N_{wells} \\cdot N_{internal}` for which the time derivative is computed by differentiating :math:`H` with respect to the complex conjugated sample :math:`\\alpha^*`.

    Args:
        * ``sample_conj``: A single complex conjugated sample.
        * ``sample``: The same sample without complex conjugation.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * The energy of the sample.
    """
    ham_singleWells = jnp.sum(
        jax.vmap(hamiltonian_singleWell, in_axes=(0, 0, None))(sample_conj, sample, cfg)
    )
    ham_jump = cfg["hamiltonianParameters"]["J"] * jnp.sum(
        (
            sample_conj * jnp.roll(sample, 1, axis=0)
            + sample * jnp.roll(sample_conj, 1, axis=0)
        )[1:, [0, 2]]
    )
    return jnp.real(ham_singleWells + ham_jump)


def hamiltonian_singleWell(sample_conj, sample, cfg):
    """
    Defines the internal hamiltonian of a single well.

    Args:
        * ``sample_conj``: A single complex conjugated sample.
        * ``sample``: The same sample without complex conjugation.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * The energy of the sample within a single well.
    """

    N = sample * sample_conj

    return (
        cfg["hamiltonianParameters"]["c_1"]
        * (
            sample_conj[0] * sample_conj[2] * sample[1] ** 2
            + sample[0] * sample[2] * sample_conj[1] ** 2
            + (N[1] - 0.5) * (N[2] + N[0] - 1)
            + 0.5
            * (
                sample_conj[0] * sample_conj[0] * sample[0] * sample[0]
                + sample_conj[2] * sample_conj[2] * sample[2] * sample[2]
                - 2 * N[2] * N[0]
            )
            + jnp.sum(N)
        )
        + cfg["hamiltonianParameters"]["c_0"] * jnp.sum(N) ** 2
        + cfg["hamiltonianParameters"]["p"] * (N[0] - N[2])
        + cfg["hamiltonianParameters"]["q"] * (N[2] + N[0])
    )
