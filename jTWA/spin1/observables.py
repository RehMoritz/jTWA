import jax
import jax.numpy as jnp


def get_spin_operators(cfg):
    """
    Returns the matrices representing the spin operators of the spin-1 system.
    The matrices are computed according to the `Jordan-Schwinger map <https://en.wikipedia.org/wiki/Jordan_map>`_.
    Note that only those matrices that are mentioned in ``cfg["simulationParameters"]["obs"]`` are returned.

    Args:
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * A dictionary containing the operators and their denotations.
    """
    Sx = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / jnp.sqrt(2)
    Sy = jnp.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / jnp.sqrt(2)
    Sz = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    S = jnp.stack((Sx, Sy, Sz))
    S_str = ["Sx", "Sy", "Sz"]

    Q = []
    Q_str = []

    for idx_1, S_1 in enumerate(S):
        for idx_2, S_2 in enumerate(S):

            if idx_1 > idx_2:
                continue

            q_ = S_1 @ S_2 + S_2 @ S_1
            if idx_1 == idx_2:
                q_ -= 4 / 3 * jnp.eye(3)

            Q.append(q_)
            Q_str.append(f"Q{S_str[idx_1][-1]}{S_str[idx_2][-1]}")
    obs = jnp.concatenate((S, -jnp.array(Q)))
    obs_str = S_str + Q_str

    idx = [i for i, o in enumerate(obs_str) if o in cfg["simulationParameters"]["obs"]]

    return {"operators": obs[idx, :], "names": [obs_str[i] for i in idx]}


def beamsplit(sample, key):
    """
    Returns a sample that is blurred with Gaussian noise.
    This recreates the effect of a broadened distribution when reading out non-commuting observables simultaneously, as for example in the case of the `Husimi Q-distribution <https://en.wikipedia.org/wiki/Husimi_Q_representation>`_.

    Args:
        * ``sample``: A single sample.
        * ``key``: A ``jax.random.PRNGKey``.

    Returns:
        * A dictionary containing the operators and their denotations.
    """
    mixer = jnp.kron(jnp.array([[1, 1], [1, -1]]), jnp.eye(3))

    noise = jax.random.normal(key, shape=(sample.shape[0], 2))
    noise = (noise[:, 0] + 1j * noise[:, 1]) / 2
    sample = jnp.concatenate((sample, noise))
    return mixer @ sample


def compute_spin_observables(operators, sample, norm):
    """
    Compute the single-well spin observables that are contained in ``operators``.
    ``sample`` is an array representing a single sample that is expected to be of shape (:math:`N_{wells}`, :math:`N_{internal}`) and ``norm`` is a normalization factor.

    Args:
        * ``operators``: An array of shape (:math:`N_{obs}`, 3, 3) corresponding to the Jordan-Schwinger matrix representations of the spin-1 operators obtained with :meth:`get_spin_operators`.
        * ``sample``: A single sample of shape (:math:`N_{wells}`, :math:`N_{internal}`) for which the operators are evaluated.
        * ``norm``: A normalization factor, usually taken to be :math:`\\sqrt{2 \\langle N \\rangle}`.

    Returns:
        * An array holding the values for each observable in each well.
        Note that these are `not` expectation values, as there is no average over all samples in this routine.
        Instead, this routine allows to obtain the full distribution of measurement outcomes.
    """
    return (
        jax.vmap(
            jax.vmap(lambda o, s: jnp.real(jnp.conj(s) @ o @ s), in_axes=(0, None)),
            in_axes=(None, 0),
        )(operators, sample)
        / norm
    )


def compute_mode_occupations(sample):
    """
    Compute the mode occupations of a sample.

    Args:
        * ``sample``: A single sample of shape (:math:`N_{wells}`, :math:`N_{internal}`) for which the mode occupations are computed.

    Returns:
        * Occupations in each mode of each well for each sample.
    """
    return jnp.abs(sample) ** 2


@jax.jit
def compute_observables(sample, key, spin_operators, norm):
    """
    Compute spin observables as well as occupations in both real space and momentum space.

    Args:
        * ``sample``: A single sample of shape (:math:`N_{wells}`, :math:`N_{internal}`).
        * ``key``: A ``jax.random.PRNGKey``, used to add Gaussian noise to the sample.
        * ``operators``: An array of shape (:math:`N_{obs}`, 3, 3) corresponding to the Jordan-Schwinger matrix representations of the spin-1 operators obtained with :meth:`get_spin_operators`.
        * ``norm``: A normalization factor, usually taken to be :math:`\\sqrt{2 \\langle N \\rangle}`.

    Returns:
        * Occupations in each mode of each well for each sample.
    """
    keys = jax.random.split(key, num=sample.shape[0])
    sample_momentumMode = jnp.fft.fft(sample, axis=0, norm="ortho")

    atom_number = compute_mode_occupations(sample)
    atom_number_momentumMode = compute_mode_occupations(sample_momentumMode)

    obs = compute_spin_observables(spin_operators, sample, norm)
    sample_split = jax.vmap(beamsplit, in_axes=(0, 0))(sample, keys)
    obs_sim = compute_spin_observables(spin_operators, sample_split[:, :3], norm)

    return {
        "atom_number_realspace": atom_number,
        "atom_number_momspace": atom_number_momentumMode,
        "spin_obs": obs,
        "spin_obs_sim": obs_sim,
    }
