import json
import pickle
import os

import jTWA


def test_visualization():
    with open(__file__.rsplit("/", 1)[0] + "/test_config.json") as f:
        cfg = json.load(f)

    with open(__file__.rsplit("/", 1)[0] + "/data.pickle", "rb") as f:
        obs = pickle.load(f)

    os.makedirs(cfg["utilParameters"]["path"])
    jTWA.visualization.plot_deviations(obs, cfg)
    jTWA.visualization.plot_spin_obs(obs, cfg)
    jTWA.visualization.plot_correlation_matrices(obs, cfg)

    assert os.path.isfile(cfg["utilParameters"]["path"] + "deviations.pdf")
    assert os.path.isfile(cfg["utilParameters"]["path"] + "spin_obs.pdf")
    assert os.path.isfile(cfg["utilParameters"]["path"] + "corrcoefs.pdf")

    os.remove(cfg["utilParameters"]["path"] + "deviations.pdf")
    os.remove(cfg["utilParameters"]["path"] + "spin_obs.pdf")
    os.remove(cfg["utilParameters"]["path"] + "corrcoefs.pdf")
    os.rmdir(cfg["utilParameters"]["path"])
    os.rmdir(cfg["utilParameters"]["path"].rsplit("/", 2)[0] + "/")
