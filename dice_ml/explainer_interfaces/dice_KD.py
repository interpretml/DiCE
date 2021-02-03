"""
Module to generate counterfactual explanations from a KD-Tree
"""
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import numpy as np
import timeit
from sklearn.neighbors import KDTree
import pandas as pd
import copy

from dice_ml import diverse_counterfactuals as exp

class DiceKD(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface) # initiating data related parameters

        # initializing model variables
        self.model = model_interface

        # loading trained model
        self.model.load_model()

        # number of output nodes of ML model
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.encoded_feature_names))

        # Partitioned dataset and KD Tree for each class (binary) of the dataset
        self.dataset_with_predictions, self.KD_tree = self.build_KD_tree()

    def build_KD_tree(self):
        # Stores the predictions on the training data
        dataset_instance = self.data_interface.prepare_query_instance(
            query_instance=self.data_interface.data_df[self.data_interface.feature_names], encoding='one-hot')
        dataset_dict_output = np.array([dataset_instance.values], dtype=np.float32)
        predictions = self.predict_fn(dataset_dict_output[0])
        # segmenting the dataset according to outcome
        dataset_with_predictions = {i: self.data_interface.data_df.loc[np.round(predictions) == i] for i in
                                    range(2)}
        # Prepares the KD trees for DiCE - 1 for each outcome (here only 0 and 1, binary classification)
        return dataset_with_predictions, {
            i: KDTree(pd.get_dummies(dataset_with_predictions[i][self.data_interface.feature_names])) for i in range(2)}

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite",
                                 feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", verbose=True):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
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

        query_instance, test_pred, final_cfs, cfs_preds = self.find_counterfactuals(query_instance, desired_class,
                                                                                    total_CFs, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, verbose)

        return exp.CounterfactualExamples(self.data_interface, query_instance, test_pred, final_cfs, cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param, desired_class)

    def predict_fn(self, input_instance):
        """prediction function"""

        temp_preds = self.model.get_output(input_instance)[:, self.num_output_nodes-1]
        return temp_preds

    def find_counterfactuals(self, query_instance, desired_class, total_CFs, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, verbose):
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

        # Finding counterfactuals from the KD Tree
        indices = self.KD_tree[desired_class].query(query_instance_df_dummies, total_CFs)[1][0].tolist()
        final_cfs = self.dataset_with_predictions[desired_class][self.data_interface.feature_names].iloc[indices].copy()

        # finding the predicted outcome for each cf
        cfs_preds = []
        for i in range(len(indices)):
            cfs = final_cfs.iloc[i].copy()
            cfs_dict = cfs.to_dict()
            cfs_dict = self.data_interface.prepare_query_instance(query_instance=cfs_dict, encoding='one-hot')
            cfs_dict = np.array([cfs_dict.iloc[0].values])
            test_pred_cf = self.predict_fn(cfs_dict)

            cfs_preds.append(test_pred_cf)

        # normalizing cfs here because we un-normalize in diverse_counterfactuals.py
        # TODO: skip this step after making relevant changes in diverse_counterfactuals.py
        final_cfs = self.data_interface.normalize_data(final_cfs)

        # Making the one-hot-encoded version of final cfs match the one-hot encoded version of the dataset
        final_cfs_dummies = pd.get_dummies(final_cfs)
        for col in self.data_interface.encoded_feature_names:
            if col not in final_cfs_dummies.columns:
                final_cfs_dummies[col] = 0

        # converting to list of arrays as required by diverse_counterfactuals.py
        rows = []
        for row in final_cfs_dummies.iterrows():
            rows.append(np.reshape(row[1].values, [1, len(self.data_interface.encoded_feature_names)]))

        final_cfs = rows

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_sparse = copy.deepcopy(final_cfs)
            cfs_preds_sparse = copy.deepcopy(cfs_preds)
            self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(total_CFs, final_cfs_sparse, cfs_preds_sparse,  query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        self.elapsed = timeit.default_timer() - start_time

        m, s = divmod(self.elapsed, 60)

        if verbose:
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')

        return query_instance, test_pred, final_cfs, cfs_preds