"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
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


class DiceGenetic(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface, model_interface)  # initiating data related parameters

        # number of output nodes of ML model
        if self.model.model_type == 'classifier':
            self.num_output_nodes = self.model.get_num_output_nodes2(
                self.data_interface.data_df[0:1][self.data_interface.feature_names])

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty

        self.population_size = 20

        # Initializing a label encoder to obtain label-encoded values for categorical variables
        self.labelencoder = {}

        self.label_encoded_data = self.data_interface.data_df.copy()

        for column in self.data_interface.categorical_feature_names:
            self.labelencoder[column] = LabelEncoder()
            self.label_encoded_data[column] = self.labelencoder[column].fit_transform(self.data_interface.data_df[column])

        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def update_hyperparameters(self, proximity_weight, diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights, encoding = 'one-hot'):
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
            if(encoding == 'one-hot'):
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            elif(encoding == 'label'):
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # TODO: why is the weight the max value of the encoded feature
                        feature_weights_list.append(self.label_encoded_data[feature].max())
            self.feature_weights_list = [feature_weights_list]

    def do_random_init(self, features_to_vary, query_instance, desired_class, desired_range):
        for kx in range(self.population_size):
            temp_cfs = []
            ix = 0
            while ix < self.total_CFs:
                one_init = [[]]
                for jx, feature in enumerate(self.data_interface.feature_names):
                    if feature in features_to_vary:
                        if feature in self.data_interface.continuous_feature_names:
                            one_init[0].append(
                                np.random.uniform(self.feature_range[feature][0], self.feature_range[feature][1]))
                        else:
                            one_init[0].append(np.random.choice(self.feature_range[feature]))
                    else:
                        one_init[0].append(query_instance[0][jx])
                if self.model.model_type == 'classifier':
                    if self.predict_fn(np.array(one_init)) != desired_class:
                        ix -= 1
                    else:
                        temp_cfs.append(np.array(one_init))
                elif self.model.model_type == 'regressor':
                    predicted_value = self.predict_fn(np.array(one_init))
                    if not desired_range[0] <= predicted_value <= desired_range[1]:
                        ix -= 1
                    else:
                        temp_cfs.append(np.array(one_init))
                ix += 1
            self.cfs.append(temp_cfs)

    def do_KD_init(self, features_to_vary, query_instance, cfs):
        cfs = self.label_encode(cfs)
        cfs = cfs.reset_index(drop=True)
        ix = 0
        done = False
        for kx in range(self.population_size):
            temp_cfs = []
            for _ in range(self.total_CFs):
                if ix >= len(cfs):
                    done = True
                    break
                one_init = [[]]
                for jx, feature in enumerate(self.data_interface.feature_names):
                    if feature not in features_to_vary:
                        one_init[0].append(query_instance[0][jx])
                    else:
                        if feature in self.data_interface.continuous_feature_names:
                            if self.feature_range[feature][0] <= cfs.iloc[ix][jx] <= self.feature_range[feature][1]:
                                one_init[0].append(cfs.iloc[ix][jx])
                            else:
                                if self.feature_range[feature][0] <= query_instance[0][jx] <= self.feature_range[feature][1]:
                                    one_init[0].append(query_instance[0][jx])
                                else:
                                    one_init[0].append(
                                        np.random.uniform(self.feature_range[feature][0],
                                                          self.feature_range[feature][1]))
                        else:
                            if cfs.iloc[ix][jx] in self.feature_range[feature]:
                                one_init[0].append(cfs.iloc[ix][jx])
                            else:
                                if query_instance[0][jx] in self.feature_range[feature]:
                                    one_init[0].append(query_instance[0][jx])
                                else:
                                    one_init[0].append(np.random.choice(self.feature_range[feature]))
                temp_cfs.append(np.array(one_init))
                ix += 1
            if done:
                break
            self.cfs.append(temp_cfs)


        for kx in range(self.population_size - len(self.cfs)):
            temp_cfs = []
            for _ in range(self.total_CFs):
                one_init = [[]]
                for jx, feature in enumerate(self.data_interface.feature_names):
                    if feature not in features_to_vary:
                        one_init[0].append(query_instance[0][jx])
                    else:
                        if feature in self.data_interface.continuous_feature_names:
                            one_init[0].append(np.random.uniform(self.feature_range[feature][0], self.feature_range[feature][1]))
                        else:
                            one_init[0].append(np.random.choice(self.feature_range[feature]))
                temp_cfs.append(np.array(one_init))
            self.cfs.append(temp_cfs)

    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, permitted_range, desired_range, desired_class, query_instance, query_instance_df_dummies, verbose):
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
            self.do_random_init(features_to_vary, query_instance, desired_class, desired_range)

        elif initialization == 'kdtree':
            # Partitioned dataset and KD Tree for each class (binary) of the dataset
            self.dataset_with_predictions, self.KD_tree, self.predictions = self.build_KD_tree(self.data_interface.data_df.copy(),
                                                                                               desired_range,
                                                                                               desired_class,
                                                                                               self.predicted_outcome_name)
            if self.KD_tree is None:
                self.do_random_init(features_to_vary, query_instance, desired_class, desired_range)

            else:
                num_queries = min(len(self.dataset_with_predictions), self.population_size*self.total_CFs)
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class, query_instance, query_instance_df_dummies, algorithm, features_to_vary, permitted_range, yloss_type, diversity_loss_type, feature_weights, proximity_weight, diversity_weight, categorical_penalty, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        self.do_cf_initializations(total_CFs, initialization, algorithm, features_to_vary, permitted_range, desired_range, desired_class, query_instance, query_instance_df_dummies, verbose)
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

    def _generate_counterfactuals(self, query_instance, total_CFs, initialization="kdtree", desired_range=None, desired_class="opposite", proximity_weight=0.5,
                                 diversity_weight=5.0, categorical_penalty=0.1, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="binary",
                                 maxiterations=10000, verbose=False):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param initialization: Method to use to initialize the population of the genetic algorithm
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: For classification problems. Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param maxiterations: Maximum iterations to run the genetic algorithm for.
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """
        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance)
        query_instance = self.label_encode(query_instance)
        query_instance = np.array([query_instance.iloc[0].values])
        self.x1 = query_instance

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)
        self.test_pred = test_pred

        if self.model.model_type == 'classifier':
            self.target_cf_class = np.array(
                [[self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)]],
                dtype=np.float32)
        elif self.model.model_type == 'regressor':
            self.target_cf_range = self.infer_target_cfs_range(desired_range)

        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in pd.get_dummies(self.data_interface.data_df[self.data_interface.feature_names]).columns:
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance, query_instance_df_dummies, algorithm, features_to_vary, permitted_range, yloss_type, diversity_loss_type, feature_weights, proximity_weight, diversity_weight, categorical_penalty, verbose)

        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, maxiterations, verbose)

        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class,
                                          model_type=self.model.model_type)

    def predict_fn_scores(self, input_instance):
        """returns predictions"""
        input_instance = self.label_decode(input_instance)
        return self.model.get_output(input_instance)

    def predict_fn(self, input_instance):
        input_instance = self.label_decode(input_instance)
        # TODO this line needs to change---we should not call model.model directly here. That functionality should be in the model class
        output = self.model.model.predict(input_instance)[0]
        return output

    def compute_yloss(self, cfs, desired_range, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.model.model_type == 'classifier':
            if self.yloss_type == 'hinge_loss':
                for i in range(self.total_CFs):
                    predicted_values = self.predict_fn_scores(cfs[i])[0]

                    maxvalue = -np.inf
                    for c in range(self.num_output_nodes):
                        if c != desired_class:
                            maxvalue = max(maxvalue, predicted_values[c])
                    temp_loss = max(0, maxvalue - predicted_values[int(desired_class)])
                    yloss += temp_loss
            return yloss/self.total_CFs

        elif self.model.model_type == 'regressor':
            if self.yloss_type == 'hinge_loss':
                for i in range(self.total_CFs):
                    predicted_value = self.predict_fn(cfs[i])
                    if desired_range[0] <= predicted_value <= desired_range[1]:
                        temp_loss = 0
                    else:
                        temp_loss = min(abs(predicted_value - desired_range[0]), abs(predicted_value - desired_range[1]))
                    yloss += temp_loss
            return yloss / self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return np.sum(np.multiply((abs(x_hat - x1)), self.feature_weights_list))

    def compute_proximity_loss(self, cfs):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(cfs[i], self.x1)
        return proximity_loss / len(self.data_interface.feature_names)

    def dpp_style(self, submethod, cfs):
        """Computes the DPP of a matrix."""
        det_entries = []
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = 1.0 / (1.0 + self.compute_dist(cfs[i], cfs[j]))
                    if i == j:
                        det_temp_entry = det_temp_entry + 0.0001
                    det_entries.append(det_temp_entry)

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = 1.0 / np.exp(
                        self.compute_dist(cfs[i], cfs[j]))
                    det_entries.append(det_temp_entry)

        det_entries = np.reshape(det_entries, [self.total_CFs, self.total_CFs])
        diversity_loss = np.linalg.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self, cfs):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return 0.0

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return np.sum(self.dpp_style(submethod, cfs))
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(cfs[i], cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self, cfs):
        """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += pow((np.sum(cfs[i][0, v[0]:v[-1] + 1]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self, cfs, desired_range, desired_class):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs) if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss(cfs) if self.diversity_weight > 0 else 0.0
        # TODO this is not needed for label encoding
        #self.regularization_loss = self.compute_regularization_loss(cfs)


        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) + (
                    self.diversity_weight * self.diversity_loss)

        return self.loss

    def mate(self, k1, k2, features_to_vary, query_instance):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        child_chromosome = []
        for i in range(self.total_CFs):
            # temp_child_chromosome = []
            one_init = [[]]
            for j in range(len(self.data_interface.feature_names)):
            #for jx, (gp1, gp2) in enumerate(zip(k1[i][0], k2[i][0])):
                gp1 = k1[i][0][j]
                gp2 = k2[i][0][j]
                feat_name = self.data_interface.feature_names[j]

                # random probability
                prob = random.random()

                # if prob is less than 0.45, insert gene from parent 1
                if prob < 0.45:
                    one_init[0].append(gp1)

                # if prob is between 0.45 and 0.90, insert gene from parent 2
                elif prob < 0.90:
                    one_init[0].append(gp2)

                #otherwise insert random gene(mutate) for maintaining diversity
                else:
                    if feat_name in features_to_vary:
                        if feat_name in self.data_interface.continuous_feature_names:
                            one_init[0].append(np.random.uniform(self.feature_range[feat_name][0], self.feature_range[feat_name][0]))
                        else:
                            one_init[0].append(np.random.choice(self.feature_range[feat_name]))
                    else:
                        one_init[0].append(query_instance[0][j])

            child_chromosome.append(np.array(one_init))
        return child_chromosome

    def find_counterfactuals(self, query_instance, desired_range, desired_class, features_to_vary, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, maxiterations, verbose):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""

        self.stopping_threshold = stopping_threshold
        if self.model.model_type == 'classifier':
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        current_best_cf = []
        stop_cnt = 0
        cfs_preds = [np.inf]*self.total_CFs
        while iterations < maxiterations:
            if abs(previous_best_loss - current_best_loss) <= 1e-2: #and (self.model.model_type == 'classifier' and all(i == desired_class for i in cfs_preds) or (self.model.model_type == 'regressor' and all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                stop_cnt += 1
            else:
                stop_cnt = 0
            if stop_cnt >= 5:
                break
            previous_best_loss = current_best_loss
            population_fitness = []
            current_best_loss = np.inf
            current_best_cf = []
            for k in range(self.population_size):
                loss = self.compute_loss(population[k], desired_range, desired_class)
                population_fitness.append((k, loss))

                if loss < current_best_loss:
                    current_best_loss = loss
                    current_best_cf = population[k]

            cfs_preds = [self.predict_fn(cfs) for cfs in current_best_cf]
            # 10% of the next generation is fittest members of current generation
            population_fitness = sorted(population_fitness, key=lambda x: x[1])
            s = int((10 * self.population_size) / 100)
            new_generation = [population[tup[0]] for tup in population_fitness[:s]]

            # 90% of the next generation obtained from top 50% of fittest members of current generation
            s = int((90 * self.population_size) / 100)
            for _ in range(s):
                parent1 = random.choice(population[:int(50 * self.population_size / 100)])
                parent2 = random.choice(population[:int(50 * self.population_size / 100)])
                child = self.mate(parent1, parent2, features_to_vary, query_instance)
                new_generation.append(child)

            population = new_generation.copy()
            iterations += 1

        self.final_cfs = current_best_cf
        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # converting to dataframe
        query_instance_df = self.label_decode(query_instance)
        query_instance_df[self.data_interface.outcome_name] = self.test_pred
        self.final_cfs_df = self.label_decode_cfs(self.final_cfs)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)
            self.final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse, query_instance_df, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            self.final_cfs_df_sparse = None

        # to display the values with the same precision as the original data
        precisions = self.data_interface.get_decimal_precisions()
        for ix, feature in enumerate(self.data_interface.continuous_feature_names):
            self.final_cfs_df[feature] = self.final_cfs_df[feature].astype(float).round(precisions[ix])
            self.final_cfs_df_sparse[feature] = self.final_cfs_df_sparse[feature].astype(float).round(precisions[ix])

        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')

        return query_instance_df

    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    def from_label(self, data):
        """Transforms label encoded data back to categorical values
        """
        out = data.copy()
        if isinstance(data, pd.DataFrame) or isinstance(data, dict):
            for column in self.categorical_feature_names:
                out[column] = self.labelencoder[column].inverse_transform(out[column].round().astype(int).tolist())
        elif isinstance(data, list):
            # TODO: make sure that the indexes match the labelencoder and list
            for c in self.categorical_feature_indexes:
                out[c] = self.labelencoder[self.feature_names[c]].inverse_transform([round(out[c])])[0]
        return out

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        labelled_input = labelled_input[0]
        input_instance = {}
        for i in range(len(labelled_input)):
            if self.data_interface.feature_names[i] in self.data_interface.categorical_feature_names:
                enc = self.labelencoder[self.data_interface.feature_names[i]]
                val = enc.inverse_transform(np.array([labelled_input[i]], dtype=np.int32))
                input_instance[self.data_interface.feature_names[i]] = val
            else:
                input_instance[self.data_interface.feature_names[i]] =labelled_input[i]

        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names, index=[0])
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
                ret_df = ret_df.append(df)
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
        return ret