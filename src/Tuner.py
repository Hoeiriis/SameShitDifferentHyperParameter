import numpy as np
import pandas as pd
from src.parameter_config.ParamConfig import ParamConfig
from src.suggestors.SuggestorBase import ParamLog
from src.utils import cd


class Tuner:

    def __init__(self, name, sam, param_config, suggestors, save_path, evaluators=None, param_log=None):
        """
        Tuner class, the main component of SameShitDifferentHyperparameter. It does automatic hyperparameter
        tuning
        :param name: Name of the tuner, uesd for saving purposes. Type: string
        :param sam: Useless param name. Anyway, it is the class instance that has hyperparameters to tune.
        It should have the following functions:
        sam.run: Given a hyperparameter suggestion, do its thing and return a score indicating the performance of that
                 hyperparameter suggestion.
        sam.save_model: A function for saving the model if needed. Only necessary if save_model is set to True in
                        function tune.
        sam.set_callbacks: A function for injecting callbacks, such as EarlyStopping, into the model.

        :param param_config: configuration of the parameters. Type: list of SingleParam
        :param suggestors: suggestors used for parameter suggestion. Type: list of suggestors
        :param save_path: storage path for saving trials. Type: string
        :param evaluators: skip for now
        :param param_log: log of tried parameters, default is None in which case a new param log i started.
               Type: ParamLog
        """
        self.tuner_name = name
        self.sam = sam
        self.suggestors = suggestors

        # Making rescaler dictionary
        p_config = ParamConfig()
        self.rescaler_functions, self.param_names = p_config.make_rescale_dict(param_config)

        # Starting log
        self.param_log = ParamLog(len(self.rescaler_functions), param_descriptions=self.param_names)\
            if param_log is None else param_log

        self.path = save_path

    def tune(self, stop_tuning, live_evals=True, save_model=False):

        trials = 0
        previous_param_performance = None
        while not stop_tuning(trials):
            param_suggestion = self._get_param_suggestions(previous_param_performance)
            previous_param_performance = self.sam.run(param_suggestion)
            self._save_log(save_model=save_model)

    def _save_log(self, save_model=False):

        actual = self.param_log.get_actual_params()
        unscaled = self.param_log.get_unscaled_params()
        score = self.param_log.get_score()

        # Constructing csv
        parameter_df = pd.DataFrame(data=actual)
        joined = pd.DataFrame(data=score).join(parameter_df)

        with cd(self.path):
            # Saving numpy arrays
            np.save("{}_params_actual.npy".format(self.tuner_name), actual)
            np.save("{}_params_unscaled.npy".format(self.tuner_name), unscaled)
            np.save("{}_params_scores.npy".format(self.tuner_name), score)

            # Saving csv
            heading = self.param_names
            heading.append("Score")
            joined.to_csv("{}_params_score".format(self.tuner_name), header=heading, index=True)

            if save_model:
                self.sam.save("{}_param_{}".format(self.tuner_name, len(actual)))

    def _get_param_suggestions(self, previous_param_performance=None):

        param_suggestions = []
        for suggestor in self.suggestors:
            param_suggestions.append(suggestor.suggest_parameters(previous_param_performance))

        return self._choose_param_suggestion(param_suggestions)

    def _choose_param_suggestion(self, param_suggestion_list):
        print("Warning: Choosing parameters from multiple suggestors has not been implemented yet."
              "Returning the first entry of the param_suggestion_list.")
        return param_suggestion_list[0]

