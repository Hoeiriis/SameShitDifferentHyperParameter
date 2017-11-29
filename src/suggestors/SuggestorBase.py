import numpy as np
import os
import keras

class SuggestorBase:

    def __init__(self, rescale_functions):

        # Setting rescale function information
        self._rescale_functions = rescale_functions
        self.n_param = len(rescale_functions)




