import jax
import sys
import json
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import jTWA


if __name__ == "__main__":
    configuration_file = sys.argv[1]

    with open(configuration_file) as f:
        cfg = json.load(f)

    observables = jTWA.spin1.observables.get_spin_operators(cfg)
    samples = jTWA.spin1.initState.getPolarState(cfg)

    cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)
    hamiltonian = jTWA.spin1.hamiltonian.hamiltonian

    obs = jTWA.integrate.stepper(samples, hamiltonian, observables, cfg)
    jTWA.util.store_data(obs, cfg)

    obs = jTWA.util.read_data(cfg)
    jTWA.visualization.create_visuals(obs, cfg)
    plt.show()
