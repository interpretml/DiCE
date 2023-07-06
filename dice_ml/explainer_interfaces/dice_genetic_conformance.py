"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
"""
import copy
import random
import timeit
import numpy as np
import pandas as pd
import os
from src.encoding.common import get_encoded_df, EncodingType
from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.utils.exception import UserConfigValidationException
import pm4py
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
'''
from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes
from pm4py.objects.conversion.log.factory import apply as df_to_eventLog
from pm4py.objects.log.util.general import CASE_ATTRIBUTE_PREFIX
import pm4py.objects.log.importer.csv.factory
'''
import itertools
from datetime import datetime
heuristic = 1
#import declare4py.src.declare4py.declare4py
class DiceGeneticConformance(ExplainerBase):

    def __init__(self, data_interface, model_interface,encoder=None,dataset=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface, model_interface)  # initiating data related parameters
        self.num_output_nodes = None

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''

        # Initializing a label encoder to obtain label-encoded values for categorical variables
        self.labelencoder = set()
        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def update_hyperparameters(self, proximity_weight, sparsity_weight,
                               diversity_weight, categorical_penalty,conformance_weight):
        """Update hyperparameters of the loss function"""

        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty
        self.conformance_weight = conformance_weight
    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights,
                                encoding='one-hot'):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]
        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type
        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=False)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            feature_weights_list = []
            if encoding == 'one-hot':
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            elif encoding == 'label':
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # the weight is inversely proportional to max value
                        feature_weights_list.append(round(1 / self.feature_range[feature].max(), 2))
            self.feature_weights_list = [feature_weights_list]

    def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
        remaining_cfs = np.zeros((num_inits, self.data_interface.number_of_features))
        # kx is the number of valid inits found so far
        kx = 0
        precisions = self.data_interface.get_decimal_precisions()
        while kx < num_inits:
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature in features_to_vary:
                    if feature in self.data_interface.continuous_feature_names:
                        one_init[jx] = np.round(np.random.uniform(
                            self.feature_range[feature][0], self.feature_range[feature][1]), precisions[jx])
                    else:
                        one_init[jx] = np.random.choice(self.feature_range[feature])
                else:
                    one_init[jx] = query_instance[jx]
            if self.is_cf_valid(self.predict_fn_scores(one_init.reshape(1,-1))):
                remaining_cfs[kx] = one_init
                kx += 1
        return remaining_cfs

    def do_KD_init(self, features_to_vary, query_instance, cfs, desired_class, desired_range):
        #cfs = self.label_encode(cfs)
        #cfs = pd.DataFrame(cfs,columns=self.data_interface.feature_names)
        cfs = cfs.reset_index(drop=True)
        query_instance = query_instance.reshape(-1,1)
        self.cfs = np.zeros((self.population_size, self.data_interface.number_of_features))
        for kx in range(self.population_size):
            if kx >= len(cfs):
                break
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = (query_instance[jx])
                else:
                    if feature in self.data_interface.continuous_feature_names:
                        if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if self.feature_range[feature][0] <= query_instance[jx] <= self.feature_range[feature][1]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.uniform(
                                    self.feature_range[feature][0], self.feature_range[feature][1])
                    else:
                        if float(cfs.iat[kx, jx]) in self.feature_range[feature]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if float(query_instance[jx]) in self.feature_range[feature]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.choice(self.feature_range[feature])
            self.cfs[kx] = one_init
            kx += 1

        new_array = [tuple(row) for row in self.cfs]
        uniques = np.unique(new_array, axis=0)

        if len(uniques) != self.population_size:
            remaining_cfs = self.do_random_init(
                self.population_size - len(uniques), features_to_vary, query_instance, desired_class, desired_range)
            self.cfs = np.concatenate([uniques, remaining_cfs])
    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, desired_range,
                              desired_class,
                              query_instance, query_instance_df_dummies, verbose):
        """Intializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1  # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        self.features_to_vary = features_to_vary

        # CF initialization
        self.cfs = []
        if initialization == 'random':
            self.cfs = self.do_random_init(
                self.population_size, features_to_vary, query_instance, desired_class, desired_range)

        elif initialization == 'kdtree':
            # Partitioned dataset and KD Tree for each class (binary) of the dataset
            self.dataset_with_predictions, self.KD_tree, self.predictions = \
                self.build_KD_tree(self.data_interface.data_df.copy(),
                                   desired_range, desired_class, self.predicted_outcome_name)
            if self.KD_tree is None:
                self.cfs = self.do_random_init(
                    self.population_size, features_to_vary, query_instance, desired_class, desired_range)

            else:
                num_queries = min(len(self.dataset_with_predictions), self.population_size * self.total_CFs)
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output, desired_class, desired_range)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class,
                                 query_instance, query_instance_df_dummies, algorithm, features_to_vary,
                                 permitted_range, yloss_type, diversity_loss_type, feature_weights,
                                 proximity_weight, sparsity_weight, diversity_weight, categorical_penalty,conformance_weight,encoder, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        if len(self.cfs) != total_CFs:
            self.do_cf_initializations(
                total_CFs, initialization, algorithm, features_to_vary, desired_range, desired_class,
                query_instance, query_instance_df_dummies, verbose)
        else:
            self.total_CFs = total_CFs
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, sparsity_weight, diversity_weight, categorical_penalty, conformance_weight)

    def _generate_counterfactuals(self,query_instance,total_CFs,encoder,dataset, initialization="kdtree",
                                  desired_range=None, desired_class="opposite", proximity_weight=0.5,
                                  sparsity_weight=0.5, diversity_weight=0.5, categorical_penalty=0.1,
                                  algorithm="DiverseCF", features_to_vary="all", permitted_range=None,
                                  yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", stopping_threshold=0.2, posthoc_sparsity_param=0,
                                  posthoc_sparsity_algorithm="linear", maxiterations=15, thresh=1e-2, verbose=True,conformance_weight=3,
                                  model_path=None,optimization=None,heuristic=None):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param initialization: Method to use to initialize the population of the genetic algorithm
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: For classification problems. Desired counterfactual class - can take 0 or 1.
                              Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters initialized
                                in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and
                                corresponding weights as values. Default option is "inverse_mad" where the
                                weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD)
                                of the feature's values in the training set; the weight for a categorical feature is
                                equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param maxiterations: Maximum iterations to run the genetic algorithm for.
        :param thresh: The genetic algorithm stops when the difference between the previous best loss and current
                       best loss is less than thresh
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                 (see diverse_counterfactuals.py).
        """

        if not hasattr(self.data_interface, 'data_df') and initialization == "kdtree":
            raise UserConfigValidationException(
                    "kd-tree initialization is not supported for private data"
                    " interface because training data to build kd-tree is not available.")

        self.population_size = 30 * total_CFs

        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance
        query_instance_orig = self.data_interface.prepare_query_instance(
                query_instance=query_instance_orig)
        query_instance = self.data_interface.prepare_query_instance(
                query_instance=query_instance)
        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(query_instance)

        #query_instance = self.label_encode(query_instance)
        #query_instance = pd.DataFrame(query_instance,columns=self.data_interface.feature_names)
        query_instance = np.array(query_instance.values[0])
        self.x1 = query_instance
        query_instance = query_instance.reshape(1,-1)
        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)

        self.test_pred = test_pred

        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred)

        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in self.data_interface.get_all_dummy_colnames():
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                      query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                      yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                      sparsity_weight, diversity_weight, categorical_penalty,conformance_weight,encoder, verbose)
        d4py = Declare4Py()
        #d4py.parse_decl_model(os.path.join(model_path,(dataset+'_'+str(encoder.prefix_length)+'.decl')))
        d4py.parse_decl_model(os.path.join(model_path,(dataset+'.decl')))
        activities, activations, targets = self.get_constraint_activities(d4py)
        '''
        start_time = timeit.default_timer()

        self.conformance_score, population_conformance, query_conformance = self.compute_conformance(
            query_instance, self.cfs, encoder, dataset, d4py
        )
        elapsed = timeit.default_timer() - start_time
        print('Conformance score computation time: ', elapsed)
        self.cfs = self.cfs[np.where(self.conformance_score > 0.95)]
        '''
        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                                      maxiterations, thresh, verbose,encoder,dataset,model_path,d4py,optimization,
                                                      heuristic,activities,activations,targets)
        ## change model given to this function
        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class,
                                          model_type=self.model.model_type)

    def predict_fn_scores(self, input_instance):
        """Returns prediction scores."""
        #input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance,columns=self.data_interface.feature_names)
        out = self.model.get_output(input_instance)
        if self.model.model_type == ModelTypes.Classifier and out.shape[1] == 1:
            # DL models return only 1 for binary classification
            out = np.hstack((1-out, out))
        return out

    def predict_fn(self, input_instance):
        """Returns actual prediction."""
        #input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance,columns=self.data_interface.feature_names)
        preds = self.model.get_output(input_instance, model_score=False)
        return preds

    def _predict_fn_custom(self, input_instance, desired_class):
        """Checks that the maximum predicted score lies in the desired class."""
        """The reason we do so can be illustrated by
        this example: If the predict probabilities are [0, 0.5, 0,5], the computed yloss is 0 as class 2 has the same
        value as the maximum score. sklearn's usual predict function, which implements argmax, returns class 1 instead
        of 2. This is why we need a custom predict function that returns the desired class if the maximum predict
        probability is the same as the probability of the desired class."""

        #input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance,columns=self.data_interface.feature_names)
        output = self.model.get_output(input_instance, model_score=True)
        if self.model.model_type == ModelTypes.Classifier and np.array(output).shape[1] == 1:
            # DL models return only 1 for binary classification
            output = np.hstack((1-output, output))
        desired_class = int(desired_class)
        maxvalues = np.max(output, 1)
        predicted_values = np.argmax(output, 1)

        # We iterate through output as we often call _predict_fn_custom for multiple inputs at once
        for i in range(len(output)):
            if output[i][desired_class] == maxvalues[i]:
                predicted_values[i] = desired_class

        return predicted_values

    def compute_yloss(self, cfs, desired_range, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.model.model_type == ModelTypes.Classifier:
            predicted_value = np.array(self.predict_fn_scores(cfs))
            if self.yloss_type == 'hinge_loss':
                maxvalue = np.full((len(predicted_value)), -np.inf)
                for c in range(self.num_output_nodes):
                    if c != desired_class:
                        maxvalue = np.maximum(maxvalue, predicted_value[:, c])
                yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])
            return yloss

        elif self.model.model_type == ModelTypes.Regressor:
            predicted_value = self.predict_fn(cfs)
            if self.yloss_type == 'hinge_loss':
                yloss = np.zeros(len(predicted_value))
                for i in range(len(predicted_value)):
                    if not desired_range[0] <= predicted_value[i] <= desired_range[1]:
                        yloss[i] = min(abs(predicted_value[i] - desired_range[0]),
                                       abs(predicted_value[i] - desired_range[1]))
            return yloss

    def compute_proximity_loss(self, x_hat_unnormalized, query_instance_normalized):
        """Compute weighted distance between two vectors."""
        x_hat = self.data_interface.normalize_data(x_hat_unnormalized)
        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in self.data_interface.continuous_feature_indexes])
        product = np.multiply(
            (abs(x_hat - query_instance_normalized)[:, [self.data_interface.continuous_feature_indexes]]),
            feature_weights)
        product = product.reshape(-1, product.shape[-1])
        proximity_loss = np.sum(product, axis=1)

        # Dividing by the sum of feature weights to normalize proximity loss
        return proximity_loss / sum(feature_weights)

    def compute_sparsity_loss(self, cfs):
        """Compute weighted distance between two vectors."""
        sparsity_loss = np.count_nonzero(np.asarray(cfs,dtype='int') - np.asarray(self.x1,dtype='int'), axis=1)
        return sparsity_loss / len(
            self.data_interface.feature_names)  # Dividing by the number of features to normalize sparsity loss
    def compute_filtered_loss(self,query_instance,cfs, desired_range, desired_class):
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 and len(self.data_interface.continuous_feature_indexes) > 1 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        # TODO DO ONLY ONE ROUND OF COMPUTE PERFORMANCE AND SAVE THE SCORE
        self.loss = np.reshape(np.array(self.yloss
                                        + (self.proximity_weight * self.proximity_loss) +
                                        + (self.sparsity_weight * self.sparsity_loss)
                                        # + (self.conformance_weight * 1 - self.conformance_score)
                                        )
                               , (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss
    def compute_loss(self, query_instance,cfs, desired_range, desired_class):
        """Computes the overall loss"""
        ##TODO Fix proximity loss
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 and len(self.data_interface.continuous_feature_indexes) > 1 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        #TODO DO ONLY ONE ROUND OF COMPUTE PERFORMANCE AND SAVE THE SCORE
        self.loss = np.reshape(np.array(self.yloss
                                            + (self.proximity_weight * self.proximity_loss) +
                                            + (self.sparsity_weight * self.sparsity_loss)
                                            + (self.conformance_weight * (1 - self.conformance_score))
                                             )
                                , (-1, 1))

        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss
    ### SHIFT FROM PARENT CONFORMANCE TO QUERY_INSTANCE CONFORMANCE
    #d4py.get_model_activities()
    #mate_2 represents the second heuristic, where we relax the contraints to include the targets that may occur
    #in the future
    def mate_2(self, k1, k2, features_to_vary, query_instance,encoder,d4py,activities,activations,targets):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        k1df = pd.DataFrame([k1], columns=self.data_interface.feature_names)
        k2df = pd.DataFrame([k2], columns=self.data_interface.feature_names)
        original_query_df = pd.DataFrame(np.asarray(query_instance,dtype=float),columns=self.data_interface.feature_names)
        encoder.decode(k1df)
        encoder.decode(k2df)
        encoder.decode(original_query_df)
        one_init = np.zeros(self.data_interface.number_of_features)
        prob = random.random()
        '''
        This decides randomly which parent's activities that are in the declare model to take
        parent1df = k1df[k1df.isin(activities)]
        parent2df = k2df[k2df.isin(activities)]
        if prob < 0.5:
            child = parent1df[parent1df.notnull()]
        else:
            child = parent2df[parent2df.notnull()]
        '''
        filter_query = original_query_df[original_query_df.isin(activities)]
        child = filter_query[filter_query.notnull()]
        child = [
        k1df[j][0] if k1df[j][0] in targets else k1df[j][0] if k1df[j][0] in targets
        else np.random.choice([x for x in encoder._label_dict[j].keys() if x not in activations])
        if ('prefix' in j) and (child[j].isnull().values == True)
        else child[j][0]
        for j in self.data_interface.feature_names
        ]
        child = pd.DataFrame([child], columns=self.data_interface.feature_names)
        encoder.encode(child)
        child = child.to_numpy().reshape(-1)
        child = np.array(child)
        for j in range(self.data_interface.number_of_features):
            feat_name = self.data_interface.feature_names[j]
            if 'prefix' not in feat_name:
                gp1 = k1[j]
                gp2 = k2[j]
                if prob < 0.40:
                    # if prob is less than 0.40, insert gene from parent 1
                    child[j] = gp1
                elif prob < 0.80:
                    # if prob is between 0.40 and 0.80, insert gene from parent 2
                    child[j] = gp2
                else:
                    # otherwise insert random gene(mutate) for maintaining diversity
                    if feat_name in features_to_vary:
                        if feat_name in self.data_interface.continuous_feature_names:
                            child[j] = np.random.uniform(self.feature_range[feat_name][0],
                                                         self.feature_range[feat_name][1])
                        else:
                            child[j] = np.random.choice(self.feature_range[feat_name])
                    else:
                        child[j] = query_instance[j]
            else:
                pass
        return child
    #mate_1 represents the first heuristic where we do not use the activities again, no matter whether they are a target or not
    def mate_1(self, k1, k2, features_to_vary, query_instance, encoder, d4py, activities, activations, targets):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        original_query_df = pd.DataFrame(np.asarray(query_instance,dtype=float),columns=self.data_interface.feature_names)
        k1df = pd.DataFrame([k1], columns=self.data_interface.feature_names)
        k2df = pd.DataFrame([k2], columns=self.data_interface.feature_names)
        encoder.decode(k1df)
        encoder.decode(k2df)
        one_init = np.zeros(self.data_interface.number_of_features)
        prob = random.random()
        '''
        This chose the parent
        parent1df = k1df[k1df.isin(activities)]
        parent2df = k2df[k2df.isin(activities)]
        null_parent1 = parent1df.isna().sum().sum()
        null_parent2 = parent2df.isna().sum().sum()
        if null_parent1 > null_parent2:
            child = parent2df[parent2df.notnull()]
        else:
            child = parent1df[parent1df.notnull()]
        '''
        filter_query = original_query_df[original_query_df.isin(activities)]
        child = filter_query[filter_query.notnull()]
        child = [
            k2df[j][0] if k1df[j][0] in activities else k1df[j][0] if k2df[j][0] in activities
            else np.random.choice([x for x in encoder._label_dict[j].keys() if x not in activities])
            if ('prefix' in j) and (child[j].isnull().values == True)
            else child[j][0]
            for j in self.data_interface.feature_names
        ]
        child = pd.DataFrame([child], columns=self.data_interface.feature_names)
        encoder.encode(child)
        child = child.to_numpy().reshape(-1)
        child = np.array(child)
        for j in range(self.data_interface.number_of_features):
            feat_name = self.data_interface.feature_names[j]
            if 'prefix' not in feat_name:
                gp1 = k1[j]
                gp2 = k2[j]
                if prob < 0.40:
                    # if prob is less than 0.40, insert gene from parent 1
                    child[j] = gp1
                elif prob < 0.80:
                    # if prob is between 0.40 and 0.80, insert gene from parent 2
                    child[j] = gp2
                else:
                    # otherwise insert random gene(mutate) for maintaining diversity
                    if feat_name in features_to_vary:
                        if feat_name in self.data_interface.continuous_feature_names:
                            child[j] = np.random.uniform(self.feature_range[feat_name][0],
                                                         self.feature_range[feat_name][1])
                        else:
                            child[j] = np.random.choice(self.feature_range[feat_name])
                    else:
                        child[j] = query_instance[j]
            else:
                pass
        return child
    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             features_to_vary, maxiterations, thresh, verbose,encoder,dataset,model_path,d4py,optimization,
                             heuristic,activities,activations,targets):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""
        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None


        self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        self.query_instance_normalized = self.query_instance_normalized.astype('float')
        while iterations < maxiterations and self.total_CFs > 0:
            if abs(previous_best_loss - current_best_loss) <= thresh and \
                    (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                     (self.model.model_type == ModelTypes.Regressor and
                      all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                stop_cnt += 1
            else:
                stop_cnt = 0
            if stop_cnt >= 5:
                break
            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)
            ##TODO: Add conformance checking here before computing fitness
            if optimization == 'filtering':
                '''
                if iterations == 1:
                    self.conformance_score, population_conformance, query_conformance = self.compute_conformance(
                        query_instance, population, encoder, dataset, d4py
                    )
                    population = population[np.where(self.conformance_score > 0.85)]
                '''
                if iterations % 2 == 0:
                    self.conformance_score, population_conformance, query_conformance = self.compute_conformance(
                        query_instance, population, encoder, dataset, d4py
                    )
                    population = population[np.where(self.conformance_score > 0.85)]

                population_fitness = self.compute_filtered_loss(query_instance,population, desired_range, desired_class)
            elif (optimization == 'loss_function') | (iterations > 0):
                self.conformance_score, population_conformance, query_conformance = self.compute_conformance(
                    query_instance, population, encoder, dataset, d4py
                )
                population_fitness = self.compute_loss(query_instance,population, desired_range, desired_class)
            population_fitness = population_fitness[population_fitness[:, 1].argsort()]
            current_best_loss = population_fitness[0][1]
            to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

            if self.total_CFs > 0:
                if self.model.model_type == ModelTypes.Classifier:
                    cfs_preds = self._predict_fn_custom(to_pred, desired_class)
                else:
                    cfs_preds = self.predict_fn(to_pred)

            # self.total_CFS of the next generation obtained from the fittest members of current generation
            top_members = self.total_CFs
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

            # rest of the next generation obtained from top 50% of fittest members of current generation
            rest_members = self.population_size - top_members
            new_generation_2 = None
            if rest_members > 0:
                new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
                for new_gen_idx in range(rest_members):
                    idx1 = random.randint(0,int(len(population) / 2))
                    parent1 = population[idx1]
                    idx2 = random.randint(0,int(len(population) / 2))
                    parent2 = population[idx2]
                    if heuristic == 'heuristic_1':
                        child = self.mate_1(parent1, parent2, features_to_vary, query_instance,encoder,d4py,activities,activations,targets)
                    elif heuristic == 'heuristic_2':
                        child = self.mate_2(parent1, parent2, features_to_vary, query_instance,encoder,d4py,activities,activations,targets)
                    new_generation_2[new_gen_idx] = child

            if new_generation_2 is not None:
                #if self.total_CFs > 0:
                population = np.concatenate([new_generation_1, new_generation_2])
                #else:
                #    population = new_generation_2
            else:
                raise SystemError("The number of total_Cfs is greater than the population size!")
            iterations += 1
        if optimization == 'filtering':
            start_time = timeit.default_timer()
            self.conformance_score, population_conformance, query_conformance = self.compute_conformance(
                query_instance, population, encoder, dataset, d4py
            )
            elapsed = timeit.default_timer() - start_time
            print('Time elapsed for conformance checking: ', elapsed)

            population = population[np.where(self.conformance_score > 0.999)]

        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        while i < self.total_CFs:
            predictions = self.predict_fn_scores(population[i].reshape(1,-1))[0]
            if self.is_cf_valid(predictions):
                self.final_cfs.append(population[i])
                # checking if predictions is a float before taking the length as len() works only for array-like
                # elements. isinstance(predictions, (np.floating, float)) checks if it's any float (numpy or otherwise)
                # We do this as we take the argmax if the prediction is a vector -- like the output of a classifier
                if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                    self.cfs_preds.append(np.argmax(predictions))
                else:
                    self.cfs_preds.append(predictions)
            i += 1

        # converting to dataframe
       #query_instance_df = self.label_decode(query_instance)
        query_instance_df = pd.DataFrame(query_instance,columns=self.data_interface.feature_names)
        query_instance_df[self.data_interface.outcome_name] = self.test_pred
        self.final_cfs_df = pd.DataFrame(self.final_cfs, columns=self.data_interface.feature_names)
        self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
            self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds
            self.round_to_precision()

        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            if len(self.final_cfs) == self.total_CFs:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                      'Diverse Counterfactuals found for the given configuation, perhaps ',
                      'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                      'min %02d' % s, 'sec')

        return query_instance_df

    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}
            for i in range(len(labelled_input[j])):
                if self.data_interface.feature_names[i] in self.data_interface.categorical_feature_names:
                    enc = self.labelencoder[self.data_interface.feature_names[i]]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[self.data_interface.feature_names[i]] = val[0]
                else:
                    temp[self.data_interface.feature_names[i]] = labelled_input[j][i]
            input_instance.append(temp)
        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        return input_instance_df

    def label_decode_cfs(self, cfs_arr):
        ret_df = None
        if cfs_arr is None:
            return None
        for cf in cfs_arr:
            df = self.label_decode(cf)
            if ret_df is None:
                ret_df = df
            else:
                ret_df = pd.concat([ret_df, df])
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            #ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
            ret[feat_name] = np.array(ret[feat_name],dtype=float)
        return ret

    def compute_conformance(self,query_instance,population,encoder,dataset,d4py):
        population_df = pd.DataFrame(population, columns=self.data_interface.feature_names)
        query_instance_to_decode = pd.DataFrame(np.array(query_instance,dtype=float), columns=self.data_interface.feature_names)
        encoder.decode(query_instance_to_decode)
        encoder.decode(population_df)
        population_df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(population_df)), 1)[0] + 1)
        population_df.insert(loc=1, column='label', value=1)
        query_instance_to_decode.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(query_instance_to_decode)), 1)[0] + 1)
        query_instance_to_decode.insert(loc=1, column='label', value=1)
        long_data = pd.wide_to_long(population_df, stubnames=['prefix'], i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        long_query_instance = pd.wide_to_long(query_instance_to_decode, stubnames=['prefix'], i='Case ID',
                                              j='order', sep='_', suffix=r'\w+')
        long_query_instance_sorted = long_query_instance.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        long_data_sorted['time:timestamp'] = timestamps
        long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
        long_data_sorted.drop(columns=['order'], inplace=True)
        columns_to_rename = {'Case ID': 'case:concept:name','prefix': 'concept:name'}
        long_data_sorted.rename(columns=columns_to_rename, inplace=True)
        long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
        long_data_sorted.replace('0', 'other', inplace=True)
        long_query_instance_sorted.rename(columns=columns_to_rename, inplace=True)
        long_query_instance_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
        long_query_instance_sorted.replace('0', 'other', inplace=True)
        event_log = pm4py.convert_to_event_log(long_data_sorted)
        query_log = pm4py.convert_to_event_log(long_query_instance_sorted)
        d4py.load_xes_log(event_log)
        model_check_res = d4py.conformance_checking(consider_vacuity=True)
        d4py.load_xes_log(query_log)
        model_check_query = d4py.conformance_checking(consider_vacuity=True)
        query_patterns = {
            constraint
            for trace, patts in model_check_query.items()
            for constraint, checker in patts.items()
            if checker.state == TraceState.SATISFIED
        }

        model_check_res = {
            k: {
                constraint: checker
                for constraint, checker in v.items()
                if checker.state != TraceState.VIOLATED and constraint in query_patterns
            }
            for k, v in model_check_res.items()
        }

        conformance_score = np.array([len(v) / len(query_patterns) for  v in model_check_res.values()])
        return conformance_score,model_check_res,query_patterns

    def get_constraint_activities(self,d4py):
        activations = set()
        targets = set()
        for constraint in d4py.model.checkers:
            # str split here for binary constraints, otherwise append if supports cardinality
            if constraint['template'].supports_cardinality:
                activations.add(constraint['attributes'])
            elif constraint['template'].is_binary:
                str1, str2 = constraint['attributes'].replace(' ', '').split(',')
                activations.add(str1)
                targets.add(str2)
        targets = set([x for x in targets if x not in activations])
        activities = activations.copy()
        activities.update(targets)
        return activities, activations, targets