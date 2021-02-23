
"""
Module to generate diverse counterfactual explanations based on random sampling.
A simple implementation.
"""

from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import math
import numpy as np
import pandas as pd
import random
import timeit
import copy
from sklearn.preprocessing import LabelEncoder

from dice_ml import diverse_counterfactuals as exp


class DiceRandom(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface)  # initiating data related parameters

        self.data_interface.create_ohe_params()
        self.model = model_interface
        self.model.load_model() # loading pickled trained model if applicable
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()

    # get data-related parameters for gradient-based DiCE - minx and max for normalized continuous features

    def _generate_counterfactuals(self, query_instance, total_CFs, desired_range=None,  desired_class="opposite", permitted_range=None, features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", sample_size=1000, random_seed=17, verbose=True):
        """Generate counterfactuals by randomly sampling features.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility

        :returns: A CounterfactualExamples object that contains the dataframe
        of generated counterfactuals as an attribute.
        """
        if permitted_range is None: # use the precomputed default
            self.feature_range = self.data_interface.permitted_range
        else: # compute the new ranges based on user input
            self.feature_range = self.data_interface.get_features_range(permitted_range)

        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == "classifier":
            self.num_output_nodes = self.predict_fn(query_instance).shape[1]

        # query_instance need no transformation for generating CFs using random sampling.
        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)[0]
        if self.model.model_type == 'classifier':
            self.target_cf_class = self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)
        elif self.model.model_type == 'regressor':
            self.target_cf_range = self.infer_target_cfs_range(desired_range)
        # fixing features that are to be fixed
        self.total_CFs = total_CFs
        if features_to_vary == "all":
            self.fixed_features_values = {}
        else:
            self.fixed_features_values = {}
            for feature in self.data_interface.feature_names:
                if feature not in features_to_vary:
                    self.fixed_features_values[feature] = query_instance[feature].iloc[0]

        self.stopping_threshold = stopping_threshold
        if self.model.model_type == "classifier":
            # TODO Generalize this for multi-class
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        # get random samples for each feature independently
        start_time = timeit.default_timer()
        self.final_cfs = self.get_samples(self.fixed_features_values, self.feature_range, sampling_random_seed=random_seed, sampling_size=sample_size)
        self.cfs_preds = None

        if self.final_cfs is not None:
            self.cfs_pred_scores = self.predict_fn(self.final_cfs)
            self.final_cfs[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.cfs_pred_scores)

        # check validity of CFs
        self.final_cfs['validity'] = self.decide_cf_validity(self.cfs_pred_scores)
        self.total_cfs_found = self.final_cfs[self.final_cfs['validity']==1].shape[0]

        if self.total_cfs_found >= self.total_CFs:
            self.final_cfs = self.final_cfs[self.final_cfs['validity'] == 1].sample(n=self.total_CFs, random_state=random_seed).reset_index(drop=True)
            self.valid_cfs_found = True
        else:
            self.final_cfs = self.final_cfs[self.final_cfs['validity'] == 1].reset_index(drop=True)
            self.valid_cfs_found = False
        final_cfs_df = self.final_cfs[self.data_interface.feature_names + [self.data_interface.outcome_name]].copy()
        final_cfs_df[self.data_interface.outcome_name] = final_cfs_df[self.data_interface.outcome_name].round(3)
        self.cfs_preds = final_cfs_df[[self.data_interface.outcome_name]].values
        self.final_cfs = final_cfs_df[self.data_interface.feature_names].values
        test_instance_df = self.data_interface.prepare_query_instance(query_instance)
        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(self.get_model_output_from_scores((test_pred,)), 3))
        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse, test_instance_df, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            final_cfs_df_sparse = None

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            if verbose:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
        else:
            if self.total_cfs_found == 0 :
                print('No Counterfactuals found for the given configuation, perhaps try with different parameters...', '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) Diverse Counterfactuals found for the given configuration, perhaps try with different parameters...' % (self.total_cfs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          final_cfs_df=final_cfs_df,
                                          test_instance_df=test_instance_df,
                                          final_cfs_df_sparse = final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_class=desired_class)

    def get_samples(self, fixed_features_values, feature_range, sampling_random_seed, sampling_size):

        # first get required parameters
        precisions = self.data_interface.get_decimal_precisions(output_type="dict")

        categorical_features_frequencies = {}
        for feature in self.data_interface.categorical_feature_names:
            categorical_features_frequencies[feature] = len(self.data_interface.data_df[feature].value_counts())

        if sampling_random_seed is not None:
            random.seed(sampling_random_seed)

        samples = []
        for feature in self.data_interface.feature_names:
            if feature in fixed_features_values:
                sample = [fixed_features_values[feature]]*sampling_size
            elif feature in self.data_interface.continuous_feature_names:
                low = feature_range[feature][0]
                high = feature_range[feature][1]
                sample = self.get_continuous_samples(low, high, precisions[feature], size=sampling_size, seed=sampling_random_seed)
            else:
                if sampling_random_seed is not None:
                    random.seed(sampling_random_seed)
                sample = random.choices(feature_range[feature], k=sampling_size)

            samples.append(sample)
        samples = pd.DataFrame(dict(zip(self.data_interface.feature_names, samples))) #to_dict(orient='records')#.values
        return samples

    def get_continuous_samples(self, low, high, precision, size=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if precision == 0:
            result = np.random.randint(low, high+1, size).tolist()
            result = [float(r) for r in result]
        else:
            result = np.random.uniform(low, high+(10**-precision), size)
            result = [round(r, precision) for r in result]
        return result

    def predict_fn(self, input_instance):
        """prediction function"""
        return self.model.get_output(input_instance)
