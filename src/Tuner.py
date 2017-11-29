import numpy as np
from src.parameter_config.ParamConfig import ParamConfig


class Tuner:

    def __init__(self, param_config):

        # Making rescaler dictionary
        p_config = ParamConfig()
        self.rescaler_functions_dict = p_config.make_rescale_dict(param_config)