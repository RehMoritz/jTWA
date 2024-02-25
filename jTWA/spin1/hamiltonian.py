import jax
import jax.numpy as jnp


def update_cfg(cfg):
    """
    Since the energy scale is set by :math:`\\langle N \\rangle c_1 = -1` we have :math:`c_1=-1/\\langle N \\rangle`.
    This computation is done at runtime to minimize the risk of inconsistencies, when setting :math:`c_1` and :math:`\\langle N \\rangle` simultaneously.
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


def hamiltonian(a_conj, a, cfg):
    """
    Defines the hamiltonian of the full system.
    The function takes both the sample and its complex conjugate as input, so that it is easy to obtain its time derivative

    .. math ::

        \\partial_t \\alpha = \\partial_{\\alpha^*} H(\\alpha, \\alpha^*).

    Args:
        * ``a_conj``: A single complex conjugated sample.
        * ``a``: The same sample without complex conjugation.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * The energy of the sample.
    """
    ham_singleWells = jnp.sum(
        jax.vmap(hamiltonian_singleWell, in_axes=(0, 0, None))(a_conj, a, cfg)
    )
    ham_jump = cfg["hamiltonianParameters"]["J"] * jnp.sum(
        (a_conj * jnp.roll(a, 1, axis=0) + a * jnp.roll(a_conj, 1, axis=0))[1:, [0, 2]]
    )
    return jnp.real(ham_singleWells + ham_jump)


def hamiltonian_singleWell(a_conj, a, cfg):
    """
    Defines the internal hamiltonian of a single well.

    Args:
        * ``a_conj``: A single complex conjugated sample.
        * ``a``: The same sample without complex conjugation.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * The energy of the sample within a single well.
    """

    N = a * a_conj

    return (
        cfg["hamiltonianParameters"]["c_1"]
        * (
            a_conj[0] * a_conj[2] * a[1] ** 2
            + a[0] * a[2] * a_conj[1] ** 2
            + (N[1] - 0.5) * (N[2] + N[0] - 1)
            + 0.5
            * (
                a_conj[0] * a_conj[0] * a[0] * a[0]
                + a_conj[2] * a_conj[2] * a[2] * a[2]
                - 2 * N[2] * N[0]
            )
            + jnp.sum(N)
        )
        + cfg["hamiltonianParameters"]["c_0"] * jnp.sum(N) ** 2
        + cfg["hamiltonianParameters"]["p"] * (N[0] - N[2])
        + cfg["hamiltonianParameters"]["q"] * (N[2] + N[0])
    )
