import pytest
import numpy as np
from src.parameter_config.ParamConfig import ParamConfig, SingleParam
from src.suggestors import RandomSearch

def test_random_search():

    # Initializing param configuration
    param_config = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0001, 0.01), scaling="log"),
        SingleParam("dropout_l", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.05),
        SingleParam("hidden_size_l", "integer", (100, 1000), "incremental", 50),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512, 1024])
    )

    p_configurer = ParamConfig()

    # Make rescaler functions
    rescaler_functions = p_configurer.make_rescale_dict(param_config)

    functions = []
    names = []

    for entry in rescaler_functions:
        names.append(entry[0])
        functions.append(entry[1])

    # Initialize RandomSearch with rescaler functions
    rand_search = RandomSearch(functions, names)

    # Making parameter suggestions
    parameter_suggestions = []
    for i in range(0, 100):
        print("Suggesting parameter: {}".format(i))
        parameter_suggestions.append(rand_search.suggest_parameters()[1])

    parameter_suggestions = np.vstack(parameter_suggestions)

    print(parameter_suggestions[:, 1])
    print(parameter_suggestions[:, 1] > 0.7)

    assert not np.any(parameter_suggestions[:, 0] < 0.0001)
    assert not np.any(parameter_suggestions[:, 0] > 0.01)

    assert not np.any(parameter_suggestions[:, 1] < 0)
    assert not np.any(parameter_suggestions[:, 1] > 0.7)

    assert not np.any(parameter_suggestions[:, 2] < 100)
    assert not np.any(parameter_suggestions[:, 2] > 1000)

    assert np.all(np.isin(parameter_suggestions[:, 3], [64, 128, 256, 512, 1024]))

