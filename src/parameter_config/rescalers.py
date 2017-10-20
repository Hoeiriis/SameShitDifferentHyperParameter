import numpy as np


def create_min_max_rescaler(original_min_max, target_min_max):
    o_min = min(original_min_max)
    o_max = max(original_min_max)
    t_min = min(target_min_max)
    t_max = max(target_min_max)

    def min_max_rescaler(x):
        return ((o_max - o_min) * (x - t_min)) / (t_max - t_min) + o_min

    return min_max_rescaler


def incremental_rescaler(incremental, min_max_range):
    min_range = min(min_max_range)
    max_range = max(min_max_range)

    # create min_max_scaler
    min_max_scaler = create_min_max_rescaler((min_range, max_range), (0, 1))

    def rescaler(value):
        # rescale value
        rescaled_value = min_max_scaler(value)
        # Round to neares incremental value and return it
        result = rescaled_value + incremental / 2
        result -= result % incremental
        return rescaled_value

    return rescaler


def log_rescaler(min_max_range):
    min_range = min(min_max_range)
    max_range = max(min_max_range)

    # create min max scaler
    min_max_scaler = create_min_max_rescaler((min_range, max_range), (0, 1))

    def rescaler(value):
        # calculating lower log limit
        a = np.log10(min_range / max_range)

        # calculate log rescale value
        r = - a * value
        log_rescale_value = 10 ** r

        return min_max_scaler(log_rescale_value)

    return rescaler
