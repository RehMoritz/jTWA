import os
import json
import pickle


def write_data(obs, cfg):
    """
    Write the computed observables to a `pickle file <https://docs.python.org/3/library/pickle.html>`_ in the dictionary that is specified in ``cfg``.

    Args:
        * ``obs``: A dictionary holding all observables that are to be stored.
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    """
    try:
        os.makedirs(cfg["utilParameters"]["path"])
    except OSError:
        print("Creation of the directory %s failed" % cfg["utilParameters"]["path"])

    with open(cfg["utilParameters"]["path"] + "data.pickle", "wb") as f:
        pickle.dump(obs, f)

    with open(cfg["utilParameters"]["path"] + "config.json", "w") as f:
        json.dump(cfg, f, indent=4)


def read_data(cfg):
    """
    Read the observables that are stored in the specified folder within ``cfg`` in pickle format.

    Args:
        * ``cfg``: The dictionary that contains the settings of the current run, including the working directory.
    Returns:
        * ``obs``: The dictionary of stored observables.
    """
    with open(cfg["utilParameters"]["path"] + "data.pickle", "rb") as f:
        obs = pickle.load(f)
    return obs
