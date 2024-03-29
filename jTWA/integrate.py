import jax
import jax.numpy as jnp
from functools import partial

import jTWA


def check_obs_save_condition(t, cfg, eps=1e-1):
    """
    Check whether observables should be computed and stored at the current time step.

    Args:
        * ``t``: The current simulation time.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * A boolean indicating whether observables should be computed and stored or not.
    """
    n = round(t / cfg["simulationParameters"]["dt_obs"])
    return (
        jnp.abs(t - n * cfg["simulationParameters"]["dt_obs"])
        < eps * cfg["simulationParameters"]["dt"]
    )


def append_observables(
    key_to_use, samples, hamiltonian, spin_operators, stored_observables, t, cfg
):
    """
    Append observables to an existing array of observables in ``stored_observables``.

    Args:
        * ``key_to_use``: A ``jax.random.PRNGKey`` that is used to mix each sample with Gaussian noise to simulate simultaneous readout.
        * ``samples``: A list of samples. The array should be of shape (:math:`N_{samples}`, :math:`N_{wells}`, :math:`N_{internal}`).
        * ``hamiltonian``: The hamiltonian function that describes the system. Used to obtain energy expectation values.
        * ``spin_operators``: A dictionary of the matrices that describe spin observables of interest along with their names.
        * ``stored_observables``: A dictionary that is either empty or contains observable values from previous time steps.
        * ``t``: The current simulation time.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * ``stored_observables``: The dictionary to which the new observables have been added.
    """
    observables_values = jax.vmap(
        jTWA.spin1.observables.compute_observables, in_axes=(0, 0, None, None)
    )(
        samples,
        jax.random.split(key_to_use, num=samples.shape[0]),
        spin_operators["operators"],
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


def integrate_single_sample(sample, flow, dt):
    """
    4-th order RK-integrator that propagates a single sample along its flow by the time step ``dt``.

    Args:
        * ``sample``: A single sample. The array should be of shape (:math:`N_{wells}`, :math:`N_{internal}`).
        * ``flow``: The flow that is induced by the Hamiltonian.
        * ``dt``: A small time step.

    Returns:
        * ``sample``: The propagated sample.
    """
    k1 = -1j * flow(jnp.conj(sample), sample)
    k2 = -1j * flow(
        jnp.conj(sample) + dt * 0.5 * jnp.conj(k1),
        sample + dt * 0.5 * k1,
    )
    k3 = -1j * flow(
        jnp.conj(sample) + dt * 0.5 * jnp.conj(k2),
        sample + dt * 0.5 * k2,
    )
    k4 = -1j * flow(
        jnp.conj(sample) + dt * jnp.conj(k3),
        sample + dt * k3,
    )
    der = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return sample + dt * der


@partial(jax.jit, static_argnums=(1,))
def integrate(samples, flow, dt):
    """
    (Compiled) Integrator that propagates the samples along their flow lines by the time step ``dt`` using a 4-th order RK-scheme described in :meth:`integrate_single_sample`.

    Args:
        * ``samples``: A list of samples. The array should be of shape (:math:`N_{samples}`, :math:`N_{wells}`, :math:`N_{internal}`).
        * ``flow``: The flow that is induced by the Hamiltonian.
        * ``dt``: A small time step.

    Returns:
        * ``samples``: The propagated samples.
    """
    samples = jax.vmap(integrate_single_sample, in_axes=(0, None, None))(
        samples, flow, dt
    )
    return samples


def obtain_evolution(samples, hamiltonian, spin_operators, cfg):
    """
    Stepper functionality to integrate the coupled equations of motion given by the Hamiltonian to the final time specified in ``cfg``.
    First, the flow that is generated by the Hamiltonian is computed.
    Then, the samples are integrated along their flow lines using :meth:`jTWA.integrate.integrate()`.
    At times of interest that are separated by the interval ``cfg["simulationParameters"]["dtObs"]``, observables are read out and stored to a dictionary.
    This dictionary is returned once the final integration time is reached.

    Args:
        * ``samples``: A list of samples. The array should be of shape (:math:`N_{samples}`, :math:`N_{wells}`, :math:`N_{internal}`).
        * ``hamiltonian``: A function that returns the energy of a given single sample, given its complex conjugate, the sample itself as well as ``cfg``.
        * ``spin_operators``: A dictionary of the matrices that describe spin observables of interest along with their names.
        * ``cfg``: The dictionary that contains the settings of the current run.

    Returns:
        * ``obs``: Dictionary that holds observables at specified times of interest.
    """

    t = 0
    key = jax.random.PRNGKey(cfg["simulationParameters"]["random_seed"])

    flow = jax.grad(partial(hamiltonian, cfg=cfg), argnums=0)

    stored_observables = {
        "spin_obs_names": spin_operators["names"],
    }

    while t < cfg["simulationParameters"]["t_end"]:

        if check_obs_save_condition(t, cfg):
            key, key_to_use = jax.random.split(key)
            stored_observables = append_observables(
                key_to_use,
                samples,
                hamiltonian,
                spin_operators,
                stored_observables,
                t,
                cfg,
            )

            print(f"t = {t:.2f}")

        samples = integrate(samples, flow, cfg["simulationParameters"]["dt"])
        t = t + cfg["simulationParameters"]["dt"]

    return stored_observables
