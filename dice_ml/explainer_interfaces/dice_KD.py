"""
Module to generate counterfactual explanations from a KD-Tree
This code is similar to 'Interpretable Counterfactual Explanations Guided by Prototypes': https://arxiv.org/pdf/1907.02584.pdf
"""
import copy
import timeit

import numpy as np
import pandas as pd

from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class DiceKD(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """
        self.total_random_inits = 0
        super().__init__(data_interface)  # initiating data related parameters

        # initializing model variables
        self.model = model_interface
        self.model.load_model()  # loading pickled trained model if applicable
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()

        # As DiCE KD uses one-hot-encoding
        # temp data to create some attributes like encoded feature names
        temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        self.data_interface.create_ohe_params(temp_ohe_data)

        # loading trained model
        self.model.load_model()

        # number of output nodes of ML model
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(
                self.data_interface.data_df[0:1][self.data_interface.feature_names])

        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def _generate_counterfactuals(self, query_instance, total_CFs, desired_range=None, desired_class="opposite",
                                  features_to_vary="all",
                                  permitted_range=None, sparsity_weight=1,
                                  feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="linear", verbose=False, limit_steps_ls=10000):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the
                              outcome class of query_instance for binary classification.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in
                                list as values. Defaults to the range inferred from training data.
                                If None, uses the parameters initialized in data_interface.
        :param sparsity_weight: Parameter to determine how much importance to give to sparsity
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding
                                weights as values. Default option is "inverse_mad" where the weight for a continuous
                                feature is the inverse of the Median Absolute Devidation (MAD) of the feature's
                                values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large (for instance, income
                                           varying from 10k to 1000k) and only if the features share a monotonic
                                           relationship with predicted outcome in the model.
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                 (see diverse_counterfactuals.py).
        """
        data_df_copy = self.data_interface.data_df.copy()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance.copy()
        query_instance_orig = self.data_interface.prepare_query_instance(
                query_instance=query_instance_orig)
        query_instance = self.data_interface.prepare_query_instance(
                query_instance=query_instance)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)[0]

        query_instance[self.data_interface.outcome_name] = test_pred
        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred)

        if desired_class == "opposite" and self.model.model_type == ModelTypes.Classifier:
            if self.num_output_nodes == 2:
                desired_class = 1.0 - test_pred

            elif self.num_output_nodes > 2:
                raise ValueError("Desired class can't be opposite if the number of classes is more than 2.")

        if isinstance(desired_class, int) and desired_class > self.num_output_nodes - 1:
            raise ValueError("Desired class should be within 0 and num_classes-1.")

        # Partitioned dataset and KD Tree for each class (binary) of the dataset
        self.dataset_with_predictions, self.KD_tree, self.predictions = \
            self.build_KD_tree(data_df_copy, desired_range, desired_class, self.predicted_outcome_name)
        query_instance, cfs_preds = self.find_counterfactuals(data_df_copy,
                                                              query_instance, query_instance_orig,
                                                              desired_range,
                                                              desired_class,
                                                              total_CFs, features_to_vary,
                                                              permitted_range,
                                                              sparsity_weight,
                                                              stopping_threshold,
                                                              posthoc_sparsity_param,
                                                              posthoc_sparsity_algorithm,
                                                              verbose,
                                                              limit_steps_ls)
        self.cfs_preds = cfs_preds

        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          final_cfs_df=self.final_cfs_df,
                                          test_instance_df=query_instance,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class,
                                          model_type=self.model.model_type)

    def predict_fn(self, input_instance):
        """returns predictions"""
        return self.model.get_output(input_instance, model_score=False)

    def do_sparsity_check(self, cfs, query_instance, sparsity_weight):
        cfs = cfs.assign(sparsity=np.nan, distancesparsity=np.nan)
        for index, row in cfs.iterrows():
            cnt = 0
            for column in self.data_interface.continuous_feature_names:
                if not np.isclose(row[column], query_instance[column].values[0]):
                    cnt += 1
            for column in self.data_interface.categorical_feature_names:
                if row[column] != query_instance[column].values[0]:
                    cnt += 1

            cfs.at[index, "sparsity"] = cnt

        cfs["distance"] = (cfs["distance"] - cfs["distance"].min()) / (cfs["distance"].max() - cfs["distance"].min())
        cfs["sparsity"] = (cfs["sparsity"] - cfs["sparsity"].min()) / (cfs["sparsity"].max() - cfs["sparsity"].min())
        cfs["distancesparsity"] = cfs["distance"] + sparsity_weight * cfs["sparsity"]
        cfs = cfs.sort_values(by="distancesparsity")
        cfs = cfs.drop(["distance", "sparsity", "distancesparsity"], axis=1)

        return cfs

    def vary_valid(self, KD_query_instance, total_CFs, features_to_vary, permitted_range, query_instance,
                   sparsity_weight):
        """This function ensures that we only vary features_to_vary when generating counterfactuals"""

        # TODO: this should be a user-specified parameter
        num_queries = min(len(self.dataset_with_predictions), total_CFs * 10)
        cfs = []

        if self.KD_tree is not None and num_queries > 0:
            KD_tree_output = self.KD_tree.query(KD_query_instance, num_queries)
            distances = KD_tree_output[0][0]
            indices = KD_tree_output[1][0]

            cfs = self.dataset_with_predictions.iloc[indices].copy()
            cfs['distance'] = distances
            cfs = self.do_sparsity_check(cfs, query_instance, sparsity_weight)
            cfs = cfs.drop(self.data_interface.outcome_name, axis=1)

        self.final_cfs = pd.DataFrame()
        final_indices = []
        cfs_preds = []
        total_cfs_found = 0

        # Iterating through the closest points from the KD tree and checking if any of these are valid
        if self.KD_tree is not None and total_CFs > 0:
            for i in range(len(cfs)):
                if total_cfs_found == total_CFs:
                    break
                valid_cf_found = True
                for feature in self.data_interface.feature_names:
                    if feature not in features_to_vary and cfs[feature].iat[i] != query_instance[feature].values[0]:
                        valid_cf_found = False
                        break
                    if feature in self.data_interface.continuous_feature_names:
                        if not self.feature_range[feature][0] <= cfs[feature].iat[i] <= self.feature_range[feature][1]:
                            valid_cf_found = False
                            break
                    else:
                        if not cfs[feature].iat[i] in self.feature_range[feature]:
                            valid_cf_found = False
                            break

                if valid_cf_found:
                    if not self.duplicates(cfs, final_indices.copy(), i):
                        total_cfs_found += 1
                        final_indices.append(i)
        if total_cfs_found > 0:
            self.final_cfs = cfs.iloc[final_indices]
            self.final_cfs = self.final_cfs.drop([self.predicted_outcome_name], axis=1)
            # Finding the predicted outcome for each cf
            for i in range(total_cfs_found):
                cfs_preds.append(
                    self.dataset_with_predictions.iloc[final_indices[i]][self.predicted_outcome_name])

        return self.final_cfs[:total_CFs], cfs_preds

    def duplicates(self, cfs, final_indices, i):
        final_indices.append(i)
        temp_cfs = cfs.iloc[final_indices]
        return temp_cfs.duplicated().iloc[-1]

    def find_counterfactuals(self, data_df_copy, query_instance, query_instance_orig, desired_range, desired_class,
                             total_CFs, features_to_vary, permitted_range,
                             sparsity_weight, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm,
                             verbose, limit_steps_ls):
        """Finds counterfactuals by querying a K-D tree for the nearest data points in the desired class from the dataset."""

        start_time = timeit.default_timer()

        # Making the one-hot-encoded version of query instance match the one-hot encoded version of the dataset
        query_instance_df_dummies = pd.get_dummies(query_instance_orig)

        data_df_columns = pd.get_dummies(data_df_copy[self.data_interface.feature_names]).columns
        for col in data_df_columns:
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        # Fix order of columns in the query instance. This is necessary because KD-tree treats data as a simple array
        # instead of a dataframe.
        query_instance_df_dummies = query_instance_df_dummies.reindex(columns=data_df_columns)

        self.final_cfs, cfs_preds = self.vary_valid(query_instance_df_dummies,
                                                    total_CFs,
                                                    features_to_vary,
                                                    permitted_range,
                                                    query_instance_orig,
                                                    sparsity_weight)

        total_cfs_found = len(self.final_cfs)
        if total_cfs_found > 0:
            # post-hoc operation on continuous features to enhance sparsity - only for public data
            if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
                self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs)
                self.final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(self.final_cfs_df_sparse, query_instance,
                                                                                posthoc_sparsity_param,
                                                                                posthoc_sparsity_algorithm,
                                                                                limit_steps_ls)
            else:
                self.final_cfs_df_sparse = None
        else:
            self.final_cfs_df_sparse = None

        self.final_cfs_df = self.final_cfs
        if total_cfs_found > 0:
            self.round_to_precision()

        self.elapsed = timeit.default_timer() - start_time

        m, s = divmod(self.elapsed, 60)

        if verbose:
            if total_cfs_found < total_CFs:
                self.elapsed = timeit.default_timer() - start_time
                m, s = divmod(self.elapsed, 60)
                print('Only %d (required %d) ' % (total_cfs_found, total_CFs),
                      'Diverse Counterfactuals found for the given configuation, perhaps ',
                      'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                      'min %02d' % s, 'sec')
            else:
                print('Diverse Counterfactuals found! total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return query_instance, cfs_preds
