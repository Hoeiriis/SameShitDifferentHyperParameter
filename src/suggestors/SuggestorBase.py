import numpy as np

class SuggestorBase:

    def __init__(self, rescale_functions):

        # Setting rescale function information
        self._rescale_functions = rescale_functions
        self.n_param = len(rescale_functions)


    def suggest_parameters(self, previous_param_performance=None):

        if previous_param_performance is not None:
            self.log_previous_param_performance(previous_param_performance)

        return self.calculate_suggestion()



    def log_previous_param_performance(self, previous_param_performance):


    def calculate_suggestion(self):
        raise NotImplementedError("This class is a baseclass,"
                                  " this function should be implemented in inheriting classes")