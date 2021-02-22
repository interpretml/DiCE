"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.explainer_interfaces"""

import numpy as np
import pandas as pd
import random
import timeit
import copy
from collections.abc import Iterable

from dice_ml import diverse_counterfactuals as exp
from dice_ml.counterfactual_explanations import CounterfactualExplanations

class ExplainerBase:

    def __init__(self, data_interface, model_interface=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """

        # initiating data and model related parameters
        self.data_interface = data_interface
        if model_interface is not None:
            #self.data_interface.create_ohe_params()
            self.model = model_interface
            self.model.load_model() # loading pickled trained model if applicable
            self.model.transformer.feed_data_params(data_interface)
            self.model.transformer.initialize_transform_func()
        # get data-related parameters for gradient-based DiCE - minx and max for normalized continuous features
        # self.total_random_inits = 0 # redundant

        # moved the following snippet to a method in public_data_interface
                # self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()
                #
                # # min and max for continuous features in original scale
                # flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
                # self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
                # org_minx, org_maxx = self.data_interface.get_minx_maxx(normalized=False)
                # self.cont_minx = list(org_minx[0][self.encoded_continuous_feature_indexes])
                # self.cont_maxx = list(org_maxx[0][self.encoded_continuous_feature_indexes])
                #
                # # decimal precisions for continuous features
                # self.cont_precisions = [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def generate_counterfactuals(self, query_instances, total_CFs, desired_class="opposite", permitted_range=None, features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", **kwargs):
        """Generate counterfactuals by randomly sampling features.

        :param query_instances: Input point(s) for which counterfactuals are to be generated. This can be a dataframe with one or more rows.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility

        :returns: A CounterfactualExplanations object that contains the list of
        counterfactual examples per query_instance as one of its attributes.
        """

        cf_examples_arr = []
        query_instances_list = []
        if isinstance(query_instances, pd.DataFrame):
            for ix in range(query_instances.shape[0]):
                query_instances_list.append(query_instances[ix:(ix+1)])
        elif isinstance(query_instances, Iterable):
            query_instances_list = query_instances
            #query_instances = query_instances.to_dict("records")
        for query_instance in query_instances_list:
            res = self._generate_counterfactuals(query_instance, total_CFs,
                    desired_class=desired_class,
                    permitted_range=permitted_range,
                    features_to_vary=features_to_vary,
                    stopping_threshold=stopping_threshold,
                    posthoc_sparsity_param=posthoc_sparsity_param,
                    posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
                    **kwargs)
            cf_examples_arr.append(res)
        return CounterfactualExplanations(cf_examples_list=cf_examples_arr)

    def feature_importance(self, query_instances, cf_examples_list=None, total_CFs=10, desired_class="opposite", permitted_range=None, features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", **kwargs):
        """ Estimate feature importance scores for the given inputs.

        TODO: do not return global importance if only one query instance is given.

        :param query_instances: A list of inputs for which to compute the
        feature importances. These can be provided as a dataframe.
        :param cf_examples_list: If precomputed, a list of counterfactual
        examples for every input point. If cf_examples_list is provided, then
        all the following parameters are ignored.
        :param total_CFs: The number of counterfactuals to generate per input
        (default is 10)
        :param other_parameters: These are the same as the
        generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
        the list of counterfactuals per input, local feature importances per
        input, and the global feature importance summarized over all inputs.
        """
        if cf_examples_list is None:
            cf_examples_list = self.generate_counterfactuals(query_instances, total_CFs,
                    desired_class=desired_class,
                    permitted_range=permitted_range,
                    features_to_vary=features_to_vary,
                    stopping_threshold=stopping_threshold,
                    posthoc_sparsity_param=posthoc_sparsity_param,
                    posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
                    **kwargs).cf_examples_list
        summary_importance = {} # initializes all values to 0
        local_importances = [{} for _ in range(len(cf_examples_list))]
        # Initializing importance vector
        allcols = self.data_interface.categorical_feature_names + self.data_interface.continuous_feature_names
        for col in allcols:
            summary_importance[col] = 0
        # Summarizing the found counterfactuals
        for i in range(len(cf_examples_list)):
            cf_examples = cf_examples_list[i]
            org_instance = cf_examples.test_instance_df

            if cf_examples.final_cfs_df_sparse is not None:
                df = cf_examples.final_cfs_df_sparse
            else:
                df = cf_examples.final_cfs_df
            # Initializing local importance for the ith query instance
            for col in allcols:
                local_importances[i][col] = 0
            for index, row in df.iterrows():
                for col in self.data_interface.continuous_feature_names:
                    if not np.isclose(org_instance[col].iloc[0], row[col]):
                        summary_importance[col] += 1
                        local_importances[i][col] += 1
                for col in self.data_interface.categorical_feature_names:
                    if org_instance[col].iloc[0] != row[col]:
                        summary_importance[col] += 1
                        local_importances[i][col] += 1
            for col in allcols:
                local_importances[i][col] /= (cf_examples_list[0].final_cfs_df.shape[0])
        for col in allcols:
            summary_importance[col] /= (cf_examples_list[0].final_cfs_df.shape[0]*len(cf_examples_list))
        return CounterfactualExplanations(cf_examples_list,
                local_importance=local_importances,
                summary_importance=summary_importance)


    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        return self.predict_fn(input_instance)

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: Final CFs in original user-fed format, in a pandas dataframe.
        :param query_instance: Query instance in original user-fed format, in a pandas dataframe.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        """
        # :param total_random_inits: total random initialization required for algorithm = RandomInitCF (where k CFs are generated by running a CF generation algorithm k times with random initializations.)

        # resetting index to make sure .loc works
        final_cfs_sparse = final_cfs_sparse.reset_index(drop=True)
        quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param)
        mads = self.data_interface.get_valid_mads()
        for feature in quantiles:
            quantiles[feature] = min(quantiles[feature], mads[feature])

        features_sorted = sorted(quantiles.items(), key=lambda kv: kv[1], reverse=True)
        for ix in range(len(features_sorted)):
            features_sorted[ix] = features_sorted[ix][0]
        precs = self.data_interface.get_decimal_precisions()
        decimal_prec = dict(zip(self.data_interface.continuous_feature_names, precs))

        cfs_preds_sparse = []
        for cf_ix in range(len(final_cfs_sparse)):
            current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

            for feature in features_sorted:
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
                #feat_ix = self.data_interface.continuous_feature_names.index(feature)
                diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]
                if(abs(diff) <= quantiles[feature]):
                    if posthoc_sparsity_algorithm == "linear":
                        final_cfs_sparse = self.do_linear_search(diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

                    elif posthoc_sparsity_algorithm == "binary":
                        final_cfs_sparse = self.do_binary_search(diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

            temp_preds = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
            temp_preds = np.round(temp_preds.flatten().tolist(), 3)[0]
            cfs_preds_sparse.append(temp_preds)

        final_cfs_sparse[self.data_interface.outcome_name] = np.array(cfs_preds_sparse)

        return final_cfs_sparse

    def do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred):
        """Performs a greedy linear search - moves the continuous features in CFs towards original values in query_instance greedily until the prediction class changes."""

        old_diff = diff
        change = (10**-decimal_prec[feature]) # the minimal possible change for a feature
        if self.model.model_type == 'classifier':
            while((abs(diff)>10e-4) and (np.sign(diff*old_diff) > 0) and
                  ((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or
                   (self.target_cf_class == 1 and current_pred > self.stopping_threshold))): # move until the prediction class changes
                old_val = final_cfs_sparse.iloc[cf_ix][feature]
                final_cfs_sparse.loc[cf_ix, feature] += np.sign(diff)*change
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
                old_diff = diff

                if(((self.target_cf_class == 0 and current_pred > self.stopping_threshold) or (self.target_cf_class == 1 and current_pred < self.stopping_threshold))):
                    final_cfs_sparse.loc[cf_ix, feature] = old_val
                    diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]
                    return final_cfs_sparse

                diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]

        elif self.model.model_type == 'regressor':
            while ((abs(diff) > 10e-4) and (np.sign(diff * old_diff) > 0) and
                   self.target_cf_range[0] <= current_pred <= self.target_cf_range[1]):  # move until the prediction class changes
                old_val = final_cfs_sparse.iloc[cf_ix][feature]
                final_cfs_sparse.loc[cf_ix, feature] += np.sign(diff) * change
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
                old_diff = diff

                if not self.target_cf_range[0] <= current_pred <= self.target_cf_range[1]:
                    final_cfs_sparse.loc[cf_ix, feature] = old_val
                    diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]
                    return final_cfs_sparse

                diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]

        return final_cfs_sparse

    def do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred):
        """Performs a binary search between continuous features of a CF and corresponding values in query_instance until the prediction class changes."""

        old_val = final_cfs_sparse.iloc[cf_ix][feature]
        final_cfs_sparse.loc[cf_ix, feature] = query_instance[feature].iloc[0]
        current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

        if self.model.model_type == 'classifier':
            # first check if assigning query_instance values to a CF is required.
            if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                return final_cfs_sparse
            else:
                final_cfs_sparse.loc[cf_ix, feature] = old_val

        elif self.model.model_type == 'regressor':
            # first check if assigning query_instance values to a CF is required.
            if self.target_cf_range[0] <= current_pred <= self.target_cf_range[1]:
                return final_cfs_sparse
            else:
                final_cfs_sparse.loc[cf_ix, feature] = old_val

        # move the CF values towards the query_instance
        if diff > 0:
            left = final_cfs_sparse.iloc[cf_ix][feature]
            right = query_instance[feature].iloc[0]

            while left <= right:
                current_val = left + ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.loc[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.model.model_type == 'classifier':
                    if (((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (
                            self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                        left = current_val + (10 ** -decimal_prec[feature])
                    else:
                        right = current_val - (10 ** -decimal_prec[feature])

                elif self.model.model_type == 'regressor':
                    if self.target_cf_range[0] <= current_pred <= self.target_cf_range[1]:
                        left = current_val + (10 ** -decimal_prec[feature])
                    else:
                        right = current_val - (10 ** -decimal_prec[feature])

        else:
            left = query_instance[feature].iloc[0]
            right = final_cfs_sparse.iloc[cf_ix][feature]

            while right >= left:
                current_val = right - ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.loc[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.model.model_type == 'classifier':
                    if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                        right = current_val - (10**-decimal_prec[feature])
                    else:
                        left = current_val + (10**-decimal_prec[feature])

                elif self.model.model_type == 'regressor':
                    if self.target_cf_range[0] <= current_pred <= self.target_cf_range[1]:
                        right = current_val - (10**-decimal_prec[feature])
                    else:
                        left = current_val + (10**-decimal_prec[feature])

        return final_cfs_sparse

    def check_permitted_range(self, permitted_range): # TODO: add comments as to where this is used if this function is necessary, else remove.
        """checks permitted range for continuous features"""
        if permitted_range is not None:
            if not self.data_interface.check_features_range(permitted_range):
                raise ValueError(
                    "permitted range of features should be within their original range")
            else:
                self.data_interface.permitted_range = permitted_range
                self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
                self.cont_minx = []
                self.cont_maxx = []
                for feature in self.data_interface.continuous_feature_names:
                    self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                    self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

    def check_mad_validity(self, feature_weights): # TODO: add comments as to where this is used if this function is necessary, else remove.
        """checks feature MAD validity and throw warnings"""
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

    def sigmoid(self, z): # used in VAE-based CF explainers
            return 1 / (1 + np.exp(-z))
