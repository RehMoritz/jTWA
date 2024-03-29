import os
import json

import jTWA


def test_storeandread_data():
    with open(__file__.rsplit("/", 1)[0] + "/test_config.json") as f:
        cfg = json.load(f)

        cfg = jTWA.spin1.hamiltonian.update_cfg(cfg)

    obs = {"myTestObservable": 1}

    jTWA.util.write_data(obs, cfg)
    jTWA.util.write_data(obs, cfg)
    read_obs = jTWA.util.read_data(cfg)

    assert obs == read_obs

    os.remove(cfg["utilParameters"]["path"] + "data.pickle")
    os.remove(cfg["utilParameters"]["path"] + "config.json")
    os.rmdir(cfg["utilParameters"]["path"])
    os.rmdir(cfg["utilParameters"]["path"].rsplit("/", 2)[0] + "/")
