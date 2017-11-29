from src.parameter_config.rescalers import *


class ParamConfig:

    def __init__(self):
        self._func_types = {
            "double": self._double,
            "integer": self._integer,
            "discrete": self._discrete
        }

    def make_rescale_dict(self, param_config):

        rescaler_functions = {}
        for key, value in param_config.items():
            rescaler_functions[key] = self._get_rescale_function(value)

        return rescaler_functions

    def _get_rescale_function(self, param_config_tuple):
        items = len(param_config_tuple)

        if param_config_tuple[0] in self._func_types.keys():
            args = []
            for i, entry in enumerate(param_config_tuple):
                if i == 0:
                    continue

                args.append(entry)

            rescaler = self._func_types[param_config_tuple[0]](*args)
        else:
            raise ValueError("The given func type \"{}\" is not supported".format(param_config_tuple[0]))

        return rescaler

    def _integer(self, min_max_range, scaling=None, incremental=None):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        if scaling is None:
            scaling = "incremental"

        if scaling == "incremental":
            if incremental is None:
                incremental = 1

            return incremental_rescaler(incremental, (min_range, max_range))

        elif scaling == "log":
            return log_rescaler((min_range, max_range), int_log=True)

        else:
            raise ValueError("The given scaling \"{}\" is not supported".format(scaling))

    def _double(self, min_max_range, scaling=None, incremental=None):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        if scaling is None:
            scaling = "incremental"

        if scaling == "incremental":
            if incremental is None:
                incremental = (max_range - min_range) / 100

            return incremental_rescaler(incremental, (min_range, max_range))

        elif scaling == "log":
            return log_rescaler((min_range, max_range))

        else:
            raise ValueError("The given scaling \"{}\" is not supported".format(scaling))

    def _discrete(self, discrete_values):
        min_range = 0
        max_range = len(discrete_values)

        internal_rescaler = incremental_rescaler(1, (min_range, max_range))

        def rescaler(value):
            idx = internal_rescaler(value)
            return discrete_values[int(idx)]

        return rescaler


if __name__ == "__main__":

    param_config = {
        "learning_rate": ("double", (0.0001, 0.01), "log"),
        "dropout_1": ("double", (0, 0.7), "incremental", 0.05),
        "hidden_size_1": ("integer", (100, 1000), "incremental", 50),
        "batch_size": ("discrete", [64, 128, 256, 512, 1024])
    }

    p_config = ParamConfig()
    rescaler_functions = p_config.make_rescale_dict(param_config)
    print(rescaler_functions.values())
    print("Learning rate rescaler input 0.5")
    print(rescaler_functions["batch_size"](0.55))
