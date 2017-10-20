import numpy as np
from .rescalers import *


class ParamConfig:

    def __init__(self, param_config, run_function=None):

        self.param_config = param_config

        self.scale_functions = {}

        def make_rescale_functions(self):
            raise NotImplementedError

    def _integer_type(self, range, scaling=None, incremental=None):
        min_range = min(range)
        max_range = max(range)

        if scaling is None:
            scaling = "incremental"

        if incremental is None:
            incremental = 1

    def _double(self, range, scaling=None, incremental=None):
        min_range = min(range)
        max_range = max(range)

        if scaling is None:
            scaling = "incremental"

        if (scaling == "incremental") & (incremental is None):
            incremental = (max_range - min_range) / 100

        if scaling == "incremental":
            return incremental_rescaler(incremental, (min_range, max_range))