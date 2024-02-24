import jax
import jax.numpy as jnp


def get_spin_operators(cfg):
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
    mixer = jnp.kron(jnp.array([[1, 1], [1, -1]]), jnp.eye(3))

    noise = jax.random.normal(key, shape=(sample.shape[0], 2))
    noise = (noise[:, 0] + 1j * noise[:, 1]) / 2
    sample = jnp.concatenate((sample, noise))
    return mixer @ sample


def compute_spin_observables(operators, samples, norm):
    return (
        jax.vmap(
            jax.vmap(lambda o, s: jnp.real(jnp.conj(s) @ o @ s), in_axes=(0, None)),
            in_axes=(None, 0),
        )(operators, samples)
        / norm
    )


def compute_mode_occupations(samples):
    return jnp.abs(samples) ** 2


@jax.jit
def compute_observables(samples, key, spin_operators, norm):
    keys = jax.random.split(key, num=samples.shape[0])
    samples_momentumMode = jnp.fft.fft(samples, axis=0, norm="ortho")

    atom_number = compute_mode_occupations(samples)
    atom_number_momentumMode = compute_mode_occupations(samples_momentumMode)

    obs = compute_spin_observables(spin_operators, samples, norm)
    samples_split = jax.vmap(beamsplit, in_axes=(0, 0))(samples, keys)
    obs_sim = compute_spin_observables(spin_operators, samples_split[:, :3], norm)

    return {
        "atom_number_realspace": atom_number,
        "atom_number_momspace": atom_number_momentumMode,
        "spin_obs": obs,
        "spin_obs_sim": obs_sim,
    }
