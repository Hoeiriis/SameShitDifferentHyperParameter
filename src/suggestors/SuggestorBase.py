import numpy as np


class SuggestorBase:

    def __init__(self, rescale_functions, param_log=None):

        # Setting rescale function information
        self._rescale_functions = rescale_functions
        self.n_param = len(rescale_functions)

        # Starting log
        self.param_log = ParamLog(self.n_param, rescale_functions.keys()) if param_log is None else param_log

    def suggest_parameters(self, previous_param_performance=None):

        if previous_param_performance is not None:
            self.param_log.log_score(previous_param_performance)

        return self.calculate_suggestion()

    def calculate_suggestion(self):
        raise NotImplementedError("This class is a baseclass,"
                                  " this function should be implemented in inheriting classes")


class ParamLog:

    def __init__(self, n_params, param_descriptions = None):

        self.n_params = n_params

        # Initializing logs
        self._actual_param_log = None
        self._unscaled_param_log = None
        self._score = None

        # Param descriptions
        if param_descriptions is not None:
            self.param_descriptions = param_descriptions

    def log_param(self, actual_param, unscaled_param, score):

        if self._actual_param_log is None:
            self._actual_param_log = actual_param.reshape((-1, self.n_params))
            self._unscaled_param_log = unscaled_param.reshape((-1, self.n_params))
            self._score = score.reshape((-1, 1))

            return True
        else:

            if self.find_param_log_idx(actual_param, list(range(0, self.n_params))) is not None:
                np.append(self._actual_param_log, actual_param.reshape((-1, self.n_params)), 0)
                np.append(self._unscaled_param_log, unscaled_param.reshape((-1, self.n_params)), 0)
                np.append(self._score, score.reshape((-1, 1)), 0)

                return True

        return False

    def log_score(self, score, idx=None):

        if idx is None:
            idx, _ = self._score.shape

    def find_param_log_idx(self, real_values, column_idx):

        # Finding the columns that contains the values
        columns = self._actual_param_log[:, column_idx]
        real_values = real_values.reshape((-1, len(column_idx)))

        # Finding wether or not the param logs contain the values in the columns
        if (columns == [real_values]).all(1).any():

            # Find index of the values and return the corresponding params
            idx = np.argmax(columns == real_values, axis=1)
            return idx

        return None

    def get_actual_params(self):
        return self._actual_param_log

    def get_unscaled_params(self):
        return self._unscaled_param_log