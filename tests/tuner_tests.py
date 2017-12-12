import pytest
import numpy as np
import os
from src.Tuner import Tuner
from src.parameter_config.ParamConfig import SingleParam


class sam_for_testing:

    def __init__(self):
        print("I'm sam")

    def run(self, params):
        return np.random.rand()*100


def test_tuner():

    # Initializing param configuration
    param_config = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0001, 0.01), scaling="log"),
        SingleParam("dropout_l", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.05),
        SingleParam("hidden_size_l", "integer", (100, 1000), "incremental", 50),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512, 1024])
    )

    sam = sam_for_testing()

    test_tuner = Tuner("test", sam=sam, param_config=param_config, suggestors="RandomSearch", save_path=os.getcwd())

    def stopper(trials):
        if trials > 10:
            return True

        return False

    test_tuner.tune(stopper)