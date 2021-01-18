"""
Module to generate diverse counterfactual explanations based on tensorflow 2.x
"""
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import tensorflow as tf

import numpy as np
import random
import timeit
from sklearn.neighbors import KDTree
import pandas as pd

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

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate

        # number of output nodes of ML model
        temp_input = tf.convert_to_tensor([tf.random.uniform([len(self.data_interface.encoded_feature_names)])], dtype=tf.float32)
        self.num_ouput_nodes = self.model.get_output(temp_input).shape[1]

        # Stores the predictions on the training data
        #TODO: Implement this for private dataset instance too
        dataset_dict = self.data_interface.prepare_dataset_instance(encode=True)
        dataset_dict_output = np.array([dataset_dict.values], dtype=np.float32)

        self.predictions = self.predict_fn(dataset_dict_output)
        self.predictions_vals = np.reshape(self.predictions[0], (self.predictions.shape[1],))

        #segmenting the dataset according to outcome
        #TODO: automate this for each outcome
        dataset_with_predictions_all = self.data_interface.append_predictions(self.predictions_vals)
        self.dataset_with_predictions = {}
        self.dataset_with_predictions[1] = dataset_with_predictions_all.loc[np.round(dataset_with_predictions_all[self.data_interface.outcome_name + "_pred"])==1]
        self.dataset_with_predictions[0] = dataset_with_predictions_all.loc[np.round(dataset_with_predictions_all[self.data_interface.outcome_name + "_pred"])==0]

        # Prepares the KD trees for DiCE - 1 for each outcome (here only 0 and 1)
        #TODO: make separate KD trees for each outcome
        self.KD_tree = {
            1: KDTree(pd.get_dummies(self.dataset_with_predictions[1][self.data_interface.feature_names])),
            0: KDTree(pd.get_dummies(self.dataset_with_predictions[0][self.data_interface.feature_names]))}

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all", permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear"):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.

        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """

        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            if not self.data_interface.check_features_range():
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

        if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        query_instance, test_pred = self.find_counterfactuals(query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm)

        return exp.CounterfactualExamples(self.data_interface, query_instance,
        test_pred, self.final_cfs, self.cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param, desired_class)

    def predict_fn(self, input_instance):
        """prediction function"""
        temp_preds = self.model.get_output(input_instance).numpy()
        return np.array([preds[(self.num_ouput_nodes-1):] for preds in temp_preds], dtype=np.float32)

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
            self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
            self.freezer = tf.constant([1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(len(self.minx[0]))])


        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = [[]]
                for jx in range(self.minx.shape[1]):
                    one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(tf.Variable(one_init, dtype=tf.float32))

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Finds counterfactuals by gradient-descent."""

        # Prepares user defined query_instance for DiCE.
        query_instance_ = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance_ = np.array([query_instance_.iloc[0].values])

        # find the predicted value of query_instance
        test_pred = self.predict_fn(tf.constant(query_instance_, dtype=tf.float32))[0][0]

        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)
        else:
            desired_class = np.round(test_pred)
        query_instance_copy = query_instance.copy()

        #preparing query instance for conversion to pandas dataframe
        for q in query_instance_copy:
            query_instance_copy[q] = [query_instance_copy[q]]
        query_instance_df = pd.DataFrame.from_dict(query_instance_copy)

        start_time = timeit.default_timer()

        #Making the one-hot-encoded version of query instance match the one-hot encoded version of the dataset
        query_instance_df_dummies = pd.get_dummies(query_instance_df)
        for col in pd.get_dummies(self.data_interface.data_df[self.data_interface.feature_names]).columns:
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        # Finding counterfactuals from the KD Tree
        indices = self.KD_tree[desired_class].query(query_instance_df_dummies, self.total_CFs)[1][0].tolist()
        final_cfs = self.dataset_with_predictions[desired_class][self.data_interface.feature_names].iloc[indices].copy()

        #finding the predicted outcome for each cf
        self.cfs_preds = []
        for i in range(len(indices)):
            cfs = final_cfs.iloc[i].copy()
            cfs_dict = cfs.to_dict()
            cfs_dict = self.data_interface.prepare_query_instance(query_instance=cfs_dict, encode=True)
            cfs_dict = np.array([cfs_dict.iloc[0].values])
            test_pred_cf = self.predict_fn(cfs_dict)
            test_pred_cf = np.reshape(test_pred_cf[0], (test_pred_cf.shape[1],))

            self.cfs_preds.append(test_pred_cf)

        #normalizing cfs here because we un-normalize in diverse_counterfactuals.py
        #TODO: skip this step after making relevant changes in diverse_counterfactuals.py
        final_cfs = self.data_interface.normalize_data(final_cfs)

        # Making the one-hot-encoded version of final cfs match the one-hot encoded version of the dataset
        final_cfs_dummies = pd.get_dummies(final_cfs)
        for col in self.data_interface.encoded_feature_names:
            if col not in final_cfs_dummies.columns:
                final_cfs_dummies[col] = 0

        #converting to list of arrays as required by diverse_counterfactuals.py
        rows = []
        for row in final_cfs_dummies.iterrows():
            rows.append(np.reshape(row[1].values, [1, len(self.data_interface.encoded_feature_names)]))

        self.final_cfs = rows

        #Not enhancing sparsity now
        #TODO: enhance sparsity later
        self.final_cfs_sparse = None
        self.cfs_preds_sparse = None

        self.elapsed = timeit.default_timer() - start_time

        m, s = divmod(self.elapsed, 60)

        print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')

        return query_instance_, test_pred