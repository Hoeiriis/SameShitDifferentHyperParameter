import numpy as np


class HyperParameterTuner:

    def __init__(self, param_config, run_function=None):

        self.param_config = param_config

        self.scale_functions = {}

    def make_rescale_functions(self):
        raise NotImplementedError

    def create_min_max_rescaler(self, original_min_max, target_min_max):
        o_min = min(original_min_max)
        o_max = max(original_min_max)
        t_min = min(target_min_max)
        t_max = max(target_min_max)

        def min_max_rescaler(x):
            return ((o_max - o_min)*(x - t_min))/(t_max - t_min)+o_min

        return min_max_rescaler

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
            return self._incremental_rescaler(incremental, (min_range, max_range))

    def _incremental_rescaler(self, incremental, min_max_range):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        # create min_max_scaler
        min_max_scaler = self.create_min_max_rescaler((min_range, max_range), (0, 1))

        def rescaler(value):
            # rescale value
            rescaled_value = min_max_scaler(value)
            # Round to neares incremental value and return it
            result = rescaled_value + incremental / 2
            result -= result % incremental
            return rescaled_value

        return rescaler

    def _log_rescaler(self, min_max_range):
        min_range = min(min_max_range)
        max_range = max(min_max_range)

        # create min max scaler
        min_max_scaler = self.create_min_max_rescaler((min_range, max_range), (0, 1))

        def rescaler(value):
            # calculating lower log limit
            a = np.log10(min_range/max_range)

            # calculate log rescale value
            r = -a*value
            log_rescale_value = 10**r

            return min_max_scaler(log_rescale_value)

        return rescaler



if __name__ == "__main__":

    param_config = {
        "learning_rate": ("double", (0.0001, 0.01), "log"),
        "dropout_1": ("double", [0, 0.7], "incremental", 0.05),
        "hidden_size_1": ("integer", (100, 1000), "incremental", 50),
        "batch_size": ("discrete", [64, 128, 256, 512, 1024])
    }

    tuner = HyperParameterTuner(param_config=param_config)

    a = (10, 100)
    b = (0, 1)

    rescaler = tuner.create_min_max_rescaler(a, b)

    print(int(rescaler(0.86)))
