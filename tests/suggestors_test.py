import pytest
import numpy as np
from src.parameter_config.ParamConfig import ParamConfig
from src.suggestors import RandomSearch

def test_random_search():

    # Initializing param configuration
    param_config = {
        "learning_rate": ("double", (0.0001, 0.01), "log"),
        "dropout_1": ("double", (0, 0.7), "incremental", 0.05),
        "hidden_size_1": ("integer", (100, 1000), "incremental", 50),
        "batch_size": ("discrete", [64, 128, 256, 512, 1024])
    }

    p_configurer = ParamConfig()

    # Make rescaler functions
    rescaler_functions = p_configurer.make_rescale_dict(param_config)

    # Initialize RandomSearch with rescaler functions
    rand_search = RandomSearch(rescaler_functions)

    parameter_suggestions = []

    for i in range(0, 100):
        print("Suggesting parameter: {}".format(i))
        parameter_suggestions.append(rand_search.suggest_parameters())

    parameter_suggestions = np.vstack(parameter_suggestions)

    print(parameter_suggestions[0])

    print(parameter_suggestions[:, 0] < 0.001)

    assert np.any(parameter_suggestions[:, 0] < 0.001) is False
    assert np.any(parameter_suggestions[:, 0] > 0.01) is False

    assert np.any(parameter_suggestions[:, 1] < 0) is False
    assert np.any(parameter_suggestions[:, 1] > 0.7) is False

    assert np.any(parameter_suggestions[:, 2] < 100) is False
    assert np.any(parameter_suggestions[:, 2] > 1000) is False

