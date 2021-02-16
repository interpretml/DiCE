"""
Module to generate counterfactual explanations from a KD-Tree
This code is similar to 'Interpretable Counterfactual Explanations Guided by Prototypes': https://arxiv.org/pdf/1907.02584.pdf
"""
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import numpy as np
import timeit
from sklearn.neighbors import KDTree
import pandas as pd
import copy
import random

from dice_ml import diverse_counterfactuals as exp


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

        # loading trained model
        self.model.load_model()

        # number of output nodes of ML model
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.encoded_feature_names))

        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

        # Partitioned dataset and KD Tree for each class (binary) of the dataset
        self.dataset_with_predictions, self.dataset_with_predictions_size, self.KD_tree, self.predictions = self.build_KD_tree()

    def build_KD_tree(self):
        # Stores the predictions on the training data
        dataset_instance = self.data_interface.prepare_query_instance(
            query_instance=self.data_interface.data_df[self.data_interface.feature_names], encoding='one-hot')
        dataset_dict_output = np.array([dataset_instance.values], dtype=np.float32)
        predictions = self.predict_fn(dataset_dict_output[0])
        # TODO: Is it okay to insert a column in the original dataframe with the predicted outcome? This is memory-efficient
        self.data_interface.data_df[self.predicted_outcome_name] = predictions
        # segmenting the dataset according to outcome
        dataset_with_predictions = {}
        dataset_with_predictions_size = {}
        for i in range(2):
            dataset_with_predictions[i] = self.data_interface.data_df.loc[np.round(predictions) == i].copy()
            dataset_with_predictions_size[i] = len(self.data_interface.data_df.loc[np.round(predictions) == i])

        # Prepares the KD trees for DiCE - 1 for each outcome (here only 0 and 1, binary classification)
        return dataset_with_predictions, dataset_with_predictions_size, \
               {i: KDTree(pd.get_dummies(dataset_with_predictions[i][self.data_interface.feature_names])) for i in
                range(2)}, predictions

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", features_to_vary="all",
                                 permitted_range=None, training_points_only=True, lmbd=1,
                                 feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                 posthoc_sparsity_algorithm="linear", verbose=True):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param training_points_only: Parameter to determine if the returned counterfactuals should be a subset of the training data points
        :param lmbd: Parameter to determine how much importance to give to sparsity
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).
        """

        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        if features_to_vary == 'all':
            features_to_vary = self.data_interface.feature_names

        query_instance, test_pred, final_cfs, cfs_preds = self.find_counterfactuals(query_instance, desired_class,
                                                                                    total_CFs, features_to_vary,
                                                                                    permitted_range,
                                                                                    training_points_only,
                                                                                    lmbd,
                                                                                    stopping_threshold,
                                                                                    posthoc_sparsity_param,
                                                                                    posthoc_sparsity_algorithm, verbose)

        return exp.CounterfactualExamples(self.data_interface, query_instance, test_pred, final_cfs, cfs_preds,
                                          self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param,
                                          desired_class)

    def predict_fn(self, input_instance):
        """prediction function"""
        temp_preds = self.model.get_output(input_instance)[:, self.num_output_nodes - 1]
        return temp_preds

    def do_sparsity_check(self, cfs, query_instance, lmbd):
        cfs = cfs.assign(sparsity=np.nan, distancesparsity=np.nan)
        for index, row in cfs.iterrows():
            cnt = 0
            for column in self.data_interface.continuous_feature_names:
                if not np.isclose(row[column], query_instance[column]):
                    cnt += 1
            for column in self.data_interface.categorical_feature_names:
                if row[column] != query_instance[column]:
                    cnt += 1

            cfs.at[index, "sparsity"] = cnt

        cfs["distance"] = (cfs["distance"] - cfs["distance"].min()) / (cfs["distance"].max() - cfs["distance"].min())
        cfs["sparsity"] = (cfs["sparsity"] - cfs["sparsity"].min()) / (cfs["sparsity"].max() - cfs["sparsity"].min())
        cfs["distancesparsity"] = cfs["distance"] + lmbd * cfs["sparsity"]
        cfs = cfs.sort_values(by="distancesparsity")

        return cfs

    def get_samples_eps(self, features_to_vary, eps, sample_size, cf, mads, query_instance, desired_class, cfs_needed):
        """This function generates counterfactuals in the epsilon-vicinity of a given counterfactual such that it
        varies only features_to_vary """

        cfs_found = []
        cfs_found_preds = []

        # The maximum number of counterfactuals this method will generate is sample_size
        for i in range(sample_size):
            temp_cf = {}
            for j in range(len(self.data_interface.feature_names)):
                feature = self.data_interface.feature_names[j]
                if feature in features_to_vary:
                    if feature in self.data_interface.categorical_feature_names:
                        # picking a random value for the feature if it is categorical
                        temp_cf[feature] = random.choice(self.data_interface.data_df[feature].unique ())
                    else:
                        # picking a value in the epsilon vicinity of the given counterfactual's feature value if it is continuous
                        minx = max(self.data_interface.permitted_range[feature][0], cf[feature] - eps * mads[feature])
                        maxx = min(self.data_interface.permitted_range[feature][1], cf[feature] + eps * mads[feature])
                        temp_cf[feature] = np.random.uniform(minx, maxx)
                else:
                    temp_cf[feature] = query_instance[feature]
            temp_cf = self.data_interface.prepare_query_instance(query_instance=temp_cf,
                                                                 encoding='one-hot')
            temp_cf = np.array([temp_cf.iloc[0].values])
            test_pred = self.predict_fn(temp_cf)[0]

            # if the instance generated is actually a counterfactual
            if np.round(test_pred) == desired_class:
                cfs_found.append(temp_cf)
                cfs_found_preds.append(test_pred)

            if len(cfs_found) == cfs_needed:
                return cfs_found, cfs_found_preds

        return cfs_found, cfs_found_preds

    def vary_only_features_to_vary(self, desired_class, KD_query_instance, total_CFs, features_to_vary, query_instance,
                                   training_points_only, lmbd):
        """This function ensures that we only vary features_to_vary when generating counterfactuals"""

        # sampling k^2 points closest points from the KD tree.
        # TODO: this should be a user-specified parameter
        num_queries = min(self.dataset_with_predictions_size[desired_class], total_CFs*total_CFs)
        KD_tree_output = self.KD_tree[desired_class].query(KD_query_instance, num_queries)
        distances = KD_tree_output[0][0]
        indices = KD_tree_output[1][0]

        cfs = self.dataset_with_predictions[desired_class][self.data_interface.feature_names].iloc[indices].copy()
        cfs['distance'] = distances
        cfs = self.do_sparsity_check(cfs, query_instance, lmbd)
        final_cfs = []
        final_indices = []
        cfs_preds = []
        total_cfs_found = 0

        # first, iterating through the closest points from the KD tree and checking if any of these are valid
        for i in range(len(cfs)):
            if total_cfs_found == total_CFs:
                break
            valid_cf_found = True
            for feature in self.data_interface.feature_names:
                if feature not in features_to_vary and cfs.iloc[i][feature] != query_instance[feature]:
                    valid_cf_found = False
                    break

            if valid_cf_found:
                total_cfs_found += 1
                final_indices.append(i)

        if total_cfs_found > 0:
            final_cfs_temp = cfs.iloc[final_indices].to_dict('records')
            final_cfs_temp = self.data_interface.prepare_query_instance(query_instance=final_cfs_temp,
                                                                        encoding='one-hot').values
            final_cfs = [final_cfs_temp[i, :].reshape(1, -1) for i in range(final_cfs_temp.shape[0])]

            # Finding the predicted outcome for each cf
            for i in range(total_cfs_found):
                cfs_preds.append(
                    self.dataset_with_predictions[desired_class].iloc[final_indices[i]][self.predicted_outcome_name])

        if total_cfs_found >= total_CFs or training_points_only:
            return final_cfs[:total_CFs], cfs_preds

        print(total_cfs_found, "Counterfactuals found so far. Moving on to non-training points")

        # Now, generating counterfactuals that aren't part of the training data
        i = 0
        j = 0
        start_eps = 1
        eps1 = start_eps
        eps2 = start_eps

        mads = self.data_interface.get_valid_mads()
        max_eps = self.data_interface.max_range
        # TODO: this should be a user-specified parameter
        sample_size = max(50, total_CFs*4)

        stop_method_1 = False
        stop_method_2 = False

        # This part of the code randomly samples points within a ball of epsilon around each point obtained from the KD tree.
        while (not stop_method_1) or (not stop_method_2):
            # Method 1 implements perturbations of all valid radii before proceeding to the next instance
            if not stop_method_1:
                cfs_found, cfs_found_preds = self.get_samples_eps(features_to_vary, eps1, sample_size, cfs.iloc[i],
                                                                  mads, query_instance, desired_class,
                                                                  total_CFs - total_cfs_found)
                final_cfs.extend(cfs_found)
                cfs_preds.extend(cfs_found_preds)
                total_cfs_found += len(cfs_found)

                # if total_CFs number of counterfactuals are already found, return
                if total_cfs_found == total_CFs:
                    return final_cfs, cfs_preds

                # double epsilon until it reaches the maximum value
                eps1 *= 2

                if eps1 > max_eps:
                    eps1 = start_eps
                    i += 1

                # stop method 1 when you have iterated through all instances
                if i == num_queries:
                    stop_method_1 = True

            # Method 2 implements perturbations of particular radius for all instances before doubling the radius
            if not stop_method_2:
                cfs_found, cfs_found_preds = self.get_samples_eps(features_to_vary, eps2, sample_size, cfs.iloc[j],
                                                                  mads, query_instance, desired_class,
                                                                  total_CFs - total_cfs_found)
                final_cfs.extend(cfs_found)
                cfs_preds.extend(cfs_found_preds)
                total_cfs_found += len(cfs_found)

                # if total_CFs number of counterfactuals are already found, return
                if total_cfs_found == total_CFs:
                    return final_cfs, cfs_preds

                # double epsilon when all instances have been covered
                if j == num_queries - 1:
                    j = -1
                    eps2 *= 2

                # stop method 2 when epsilon has reached the maximum value
                if eps2 > max_eps:
                    stop_method_2 = True
                j += 1

        return final_cfs, cfs_preds

    def find_counterfactuals(self, query_instance, desired_class, total_CFs, features_to_vary, permitted_range,
                             training_points_only, lmbd, stopping_threshold,
                             posthoc_sparsity_param, posthoc_sparsity_algorithm, verbose):
        """Finds counterfactuals by querying a K-D tree for the nearest data points in the desired class from the dataset."""

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encoding='one-hot')
        query_instance = np.array([query_instance.iloc[0].values])

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)[0]

        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)
        else:
            desired_class = np.round(test_pred)
        self.target_cf_class = np.array([[desired_class]], dtype=np.float32)

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        query_instance_copy = query_instance_orig.copy()

        # preparing query instance for conversion to pandas dataframe
        for q in query_instance_copy:
            query_instance_copy[q] = [query_instance_copy[q]]
        query_instance_df = pd.DataFrame.from_dict(query_instance_copy)

        start_time = timeit.default_timer()

        # Making the one-hot-encoded version of query instance match the one-hot encoded version of the dataset
        query_instance_df_dummies = pd.get_dummies(query_instance_df)
        for col in pd.get_dummies(self.data_interface.data_df[self.data_interface.feature_names]).columns:
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        final_cfs, cfs_preds = self.vary_only_features_to_vary(desired_class,
                                                               query_instance_df_dummies,
                                                               total_CFs,
                                                               features_to_vary,
                                                               query_instance_orig,
                                                               training_points_only, lmbd)

        total_cfs_found = len(final_cfs)
        if total_cfs_found > 0:
            # post-hoc operation on continuous features to enhance sparsity - only for public data
            if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
                final_cfs_sparse = copy.deepcopy(final_cfs)
                cfs_preds_sparse = copy.deepcopy(cfs_preds)
                self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(total_CFs,
                                                                                                    final_cfs_sparse,
                                                                                                    cfs_preds_sparse,
                                                                                                    query_instance,
                                                                                                    posthoc_sparsity_param,
                                                                                                    posthoc_sparsity_algorithm)
            else:
                self.final_cfs_sparse = None
                self.cfs_preds_sparse = None
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        self.elapsed = timeit.default_timer() - start_time

        m, s = divmod(self.elapsed, 60)

        if verbose:
            if total_cfs_found < total_CFs:
                self.elapsed = timeit.default_timer() - start_time
                m, s = divmod(self.elapsed, 60)
                print(
                    'Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps change the query instance or the features to vary...' % (
                        total_cfs_found, total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Diverse Counterfactuals found! total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return query_instance, test_pred, final_cfs, cfs_preds
