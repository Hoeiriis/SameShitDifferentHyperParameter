import numpy as np
from .SuggestorBase import SuggestorBase


class RandomSearch(SuggestorBase):

    def calculate_suggestion(self):
        unique = False

        while not unique:
            real, unscaled = self._random_param_sample()

            unique = self.param_log.log_param(real, unscaled, np.array([0]))

        return real

    def _random_param_sample(self):
        real_parameters = []
        unscaled_parameters = np.random.random_sample(self.n_param)

        for i, func in enumerate(self._rescale_functions.values()):
            real_parameters.append(func(unscaled_parameters[i]))

        return np.asarray(real_parameters), unscaled_parameters
