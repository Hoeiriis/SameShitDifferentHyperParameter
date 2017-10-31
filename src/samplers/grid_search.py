import numpy as np
import os
import keras

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.newPath = os.path.expanduser(new_path)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class SamplerBase:

    def __init__(self, rescale_functions, amount, results_path):

        # Setting rescale function information
        self._rescale_functions = rescale_functions
        self.n_param = len(rescale_functions)

        if amount is None:
            self._amount = 1000
        else:
            self._amount = amount

        self.results_path = results_path

    def give_feedback(self, model, history):

        if model is None:
            raise ValueError("Parameter model can not be None")

        if history is None:
            raise ValueError("Parameter history can not be None")

        raise NotImplementedError("This class is a base for sampler classes and thus has no functionality")

    def get_param(self, history=None, model=None, feedback=True):

        if feedback:
            self.give_feedback(history, model)

        raise NotImplementedError("This class is a base for sampler classes and thus has no functionality")


