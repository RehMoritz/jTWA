import jax
import jax.numpy as jnp


def update_cfg(cfg):
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
    ham_singleWells = jnp.sum(
        jax.vmap(hamiltonian_singleWell, in_axes=(0, 0, None))(a_conj, a, cfg)
    )
    ham_jump = cfg["hamiltonianParameters"]["J"] * jnp.sum(
        (a_conj * jnp.roll(a, 1, axis=0) + a * jnp.roll(a_conj, 1, axis=0))[1:, [0, 2]]
    )
    return jnp.real(ham_singleWells + ham_jump)


def hamiltonian_singleWell(a_conj, a, cfg):
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
