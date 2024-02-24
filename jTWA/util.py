import os
import json
import pickle


def store_data(obs, cfg):
    try:
        os.makedirs(cfg["utilParameters"]["path"])
    except OSError:
        print("Creation of the directory %s failed" % cfg["utilParameters"]["path"])

    with open(cfg["utilParameters"]["path"] + "data.pickle", "wb") as f:
        pickle.dump(obs, f)

    with open(cfg["utilParameters"]["path"] + "config.json", "w") as f:
        json.dump(cfg, f, indent=4)


def read_data(cfg):
    with open(cfg["utilParameters"]["path"] + "data.pickle", "rb") as f:
        obs = pickle.load(f)
    return obs
