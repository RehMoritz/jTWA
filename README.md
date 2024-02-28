# jTWA
[![Documentation Status](https://readthedocs.org/projects/jtwa/badge/?version=latest)](https://jtwa.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/github/RehMoritz/jTWA/graph/badge.svg?token=TY92JAI1V9)](https://codecov.io/github/RehMoritz/jTWA)

jTWA implements the semiclassical [Truncated Wigner Approximation (TWA)](https://www.sciencedirect.com/science/article/pii/S0003491610000382?via%3Dihub) in python, relying on Google's jax library.
This allows to write easily understandable python code without compromising on speed, as all calculations are compiled and executed on GPUs, if available.

jTWA is designed to work with bosonic systems with potentially different numbers of internal degrees of freedom.
However, basic functionalities for spin-1 systems are already available in ``jTWA.spin1``.

## Documentation & Installation
All code is documented on [readthedocs](https://jtwa.readthedocs.io/en/latest/#).
Working with jTWA is designed to be straightforward. For the installation procedure, have a look at the [corresponding section](https://jtwa.readthedocs.io/en/latest/docs/install.html) in the documentation.

## Getting Started: A Minimal Example
A minimal working example of the codebase can be found in the [``main.py``](https://github.com/RehMoritz/jTWA/blob/main/main.py) located in the root of the directory.
To execute, run `python main.py config.json` as described in the [quickstart section](https://jtwa.readthedocs.io/en/latest/docs/quickstart.html) of the documentation.
To demonstrate the elementary features, here are the contents of the [``main.py``](https://github.com/RehMoritz/jTWA/blob/main/main.py):

```python
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

    spin_operators = jTWA.spin1.observables.get_spin_operators(cfg)
    samples = jTWA.spin1.initState.getPolarState(cfg)

    cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)
    hamiltonian = jTWA.spin1.hamiltonian.hamiltonian

    obs = jTWA.integrate.obtain_evolution(samples, hamiltonian, spin_operators, cfg)
    jTWA.util.write_data(obs, cfg)

    obs = jTWA.util.read_data(cfg)
    jTWA.visualization.create_visuals(obs, cfg)
    plt.show()
```

## Issues & Requests

If you encounter any issues, please open an [issue](https://github.com/RehMoritz/jTWA/issues) or submit a [pull request](https://github.com/RehMoritz/jTWA/pulls).

## License

[Apache License 2.0](https://github.com/RehMoritz/jTWA/blob/main/LICENSE)
