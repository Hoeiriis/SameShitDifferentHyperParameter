import numpy as np
import pandas as pd
from src.parameter_config.ParamConfig import ParamConfig
from src.suggestors.SuggestorBase import ParamLog
from src.utils import cd


class Tuner:

    def __init__(self, name, sam, param_config, suggestors, save_path, evaluators=None, param_log=None):

        self.name = name
        self.sam = sam
        self.suggestors = suggestors

        # Making rescaler dictionary
        p_config = ParamConfig()
        self.rescaler_functions, self.param_names = p_config.make_rescale_dict(param_config)

        # Starting log
        self.param_log = ParamLog(len(self.rescaler_functions), param_descriptions=self.param_names)\
            if param_log is None else param_log

        self.path = save_path

    def save_log(self, save_model=False):

        actual = self.param_log.get_actual_params()
        unscaled = self.param_log.get_unscaled_params()
        score = self.param_log.get_score()

        # Constructing csv
        parameter_df = pd.DataFrame(data={'Parameters': actual})
        joined = pd.DataFrame(data={"Score": score}).join(parameter_df)

        with cd(self.path):
            # Saving numpy arrays
            np.save("{}_params_actual.npy".format(self.name), actual)
            np.save("{}_params_unscaled.npy".format(self.name), unscaled)
            np.save("{}_params_scores.npy".format(self.name), score)

            # Saving csv
            joined.to_csv("{}_params_score".format(self.name), index=False)

            if save_model:
                self.sam.save("{}_param_{}".format(self.name, len(actual)))

    def get_param_suggestions(self, previous_param_performance=None):

        param_suggestions = []
        for suggestor in self.suggestors:
            param_suggestions.append(suggestor.suggest_parameters(previous_param_performance))

        return self.choose_param_suggestion(param_suggestions)

    def choose_param_suggestion(self, param_suggestion_list):
        print("Warning: Choosing parameters from multiple suggestors has not been implemented yet."
              "Returning the first entry of the param_suggestion_list.")
        return param_suggestion_list[0]
