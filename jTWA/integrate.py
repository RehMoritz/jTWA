import jax
import jax.numpy as jnp
from functools import partial

import jTWA


def check_obs_save_condition(t, cfg, eps=1e-1):
    n = round(t / cfg["simulationParameters"]["dt_obs"])
    return (
        jnp.abs(t - n * cfg["simulationParameters"]["dt_obs"])
        < eps * cfg["simulationParameters"]["dt"]
    )


def append_observables(
    key_to_use, samples, hamiltonian, observables, stored_observables, t, cfg
):
    observables_values = jax.vmap(
        jTWA.spin1.observables.compute_observables, in_axes=(0, 0, None, None)
    )(
        samples,
        jax.random.split(key_to_use, num=samples.shape[0]),
        observables["operators"],
        jnp.sqrt(2 * cfg["systemParameters"]["n_atoms_per_well"]),
    )
    energy_exp = jnp.mean(jax.vmap(lambda s: hamiltonian(jnp.conj(s), s, cfg))(samples))
    observables_values["energy_exp"] = jnp.array([energy_exp])
    observables_values["t"] = jnp.array([t])

    for obs_key, obs_val in observables_values.items():
        try:
            stored_observables[obs_key] = jnp.concatenate(
                [stored_observables[obs_key], obs_val[None, ...]]
            )
        except KeyError:
            stored_observables[obs_key] = obs_val[None, ...]

    return stored_observables


def integrate_single_sample(sample_coord, flow, dt):
    k1 = -1j * flow(jnp.conj(sample_coord), sample_coord)
    k2 = -1j * flow(
        jnp.conj(sample_coord) + dt * 0.5 * jnp.conj(k1),
        sample_coord + dt * 0.5 * k1,
    )
    k3 = -1j * flow(
        jnp.conj(sample_coord) + dt * 0.5 * jnp.conj(k2),
        sample_coord + dt * 0.5 * k2,
    )
    k4 = -1j * flow(
        jnp.conj(sample_coord) + dt * jnp.conj(k3),
        sample_coord + dt * k3,
    )
    der = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return sample_coord + dt * der


@partial(jax.jit, static_argnums=(1,))
def integrate(samples, flow, dt):
    samples = jax.vmap(integrate_single_sample, in_axes=(0, None, None))(
        samples, flow, dt
    )
    return samples


def stepper(samples, hamiltonian, observables, cfg):
    t = 0
    key = jax.random.PRNGKey(cfg["simulationParameters"]["random_seed"])

    flow = jax.grad(partial(hamiltonian, cfg=cfg), argnums=0)

    stored_observables = {
        "spin_obs_names": observables["names"],
    }

    while t < cfg["simulationParameters"]["t_end"]:

        if check_obs_save_condition(t, cfg):
            key, key_to_use = jax.random.split(key)
            stored_observables = append_observables(
                key_to_use,
                samples,
                hamiltonian,
                observables,
                stored_observables,
                t,
                cfg,
            )

            print(f"t = {t:.2f}")

        samples = integrate(samples, flow, cfg["simulationParameters"]["dt"])
        t = t + cfg["simulationParameters"]["dt"]

    return stored_observables
