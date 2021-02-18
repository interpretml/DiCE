"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
"""

from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
import math
import numpy as np
import random
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp

class DiceGenetic(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface)  # initiating data related parameters

        # initializing model variables
        self.model = model_interface

        # loading trained model
        self.model.load_model()

        # number of output nodes of ML model
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.feature_names))

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty

        self.population_size = 100

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
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
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
                        feature_weights_list.append(self.data_interface.label_encoded_data[feature].max())
            self.feature_weights_list = [feature_weights_list]

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary, desired_class, query_instance):
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
        for kx in range(self.population_size):
            temp_cfs = []
            ix = 0
            while ix < self.total_CFs:
                one_init = [[]]
                for jx, feature in enumerate(self.data_interface.feature_names):
                    if feature in features_to_vary:
                        one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                    else:
                        one_init[0].append(query_instance[0][jx])
                if self.predict_fn(np.array(one_init)) != desired_class:
                    ix -= 1
                else:
                    temp_cfs.append(np.array(one_init))
                ix += 1
            self.cfs.append(temp_cfs)
        print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, desired_class, query_instance, algorithm, features_to_vary, yloss_type, diversity_loss_type, feature_weights, proximity_weight, diversity_weight, categorical_penalty, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True, encoding='label')
        self.do_cf_initializations(total_CFs, algorithm, features_to_vary, desired_class, query_instance)
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", proximity_weight=0.5,
                                 diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", verbose=True):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
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
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """
        self.check_mad_validity(feature_weights)
        self.check_permitted_range(permitted_range)

        # Prepares user defined query_instance for DiCE.
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encoding='label')
        query_instance = np.array([query_instance.iloc[0].values])
        self.x1 = query_instance

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)

        if desired_class == "opposite" and self.num_output_nodes == 2:
            desired_class = 1.0 - test_pred

        if desired_class == "opposite" and self.num_output_nodes > 2:
            raise ValueError("Desired class can't be opposite if the number of classes is more than 2.")

        if isinstance(desired_class, int) and desired_class > self.num_output_nodes-1:
            raise ValueError("Desired class should be within 0 and num_classes-1.")

        self.target_cf_class = np.array([[desired_class]], dtype=np.float32)

        if features_to_vary == 'all':
            features_to_vary = self.data_interface.feature_names

        self.do_param_initializations(total_CFs, desired_class, query_instance, algorithm, features_to_vary, yloss_type, diversity_loss_type, feature_weights, proximity_weight, diversity_weight, categorical_penalty, verbose)

        query_instance = self.find_counterfactuals(query_instance, desired_class, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, verbose)
        return exp.CounterfactualExamples(self.data_interface, query_instance, test_pred, self.final_cfs, self.cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param, desired_class, encoding='label')

    def predict_proba_fn(self, input_instance):
        """returns prediction probabilities"""
        return self.model.get_output(input_instance)[0]

    def predict_fn(self, input_instance):
        """returns predictions"""
        return self.model.model.predict(input_instance)[0]


    def compute_yloss(self, cfs, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.yloss_type == 'hinge_loss':
            for i in range(self.total_CFs):
                predicted_values = self.predict_proba_fn(cfs[i])

                maxvalue = -np.inf
                for c in range(self.num_output_nodes):
                    if c != desired_class:
                        maxvalue = max(maxvalue, predicted_values[c])
                temp_loss = max(0, maxvalue - predicted_values[int(desired_class)])
                yloss += temp_loss
        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return np.sum(np.multiply((abs(x_hat - x1)), self.feature_weights_list))

    def compute_proximity_loss(self, cfs):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(cfs[i], self.x1)
        return proximity_loss / len(self.minx[0])

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

    def compute_loss(self, cfs, desired_class):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss(cfs, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs) if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss(cfs) if self.diversity_weight > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss(cfs)

        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - (
                    self.diversity_weight * self.diversity_loss) + (
                                self.categorical_penalty * self.regularization_loss)

        return self.loss

    def mate(self, k1, k2):
        """Performs mating and produces new offsprings"""

        # chromosome for offspring
        child_chromosome = []
        for i in range(self.total_CFs):
            # temp_child_chromosome = []
            one_init = [[]]
            for jx, (gp1, gp2) in enumerate(zip(k1[i][0], k2[i][0])):
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
                     one_init[0].append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
            child_chromosome.append(np.array(one_init))
        return child_chromosome

    def find_counterfactuals(self, query_instance, desired_class, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, verbose):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        population = self.cfs.copy()

        start_time = timeit.default_timer()

        while True:
            population_fitness = []
            current_best_loss = np.inf
            current_best_cf = []
            for k in range(self.population_size):
                loss = self.compute_loss(population[k], desired_class)
                population_fitness.append((k, loss))

                if loss < current_best_loss:
                    current_best_loss = loss
                    current_best_cf = population[k]

            pop_pred = [self.predict_fn(cfs) for cfs in current_best_cf]

            if all(i == desired_class for i in pop_pred):
                self.valid_cfs_found = True
                break

            # 10% of the next generation is fittest members of current generation
            population_fitness = sorted(population_fitness, key=lambda x: x[1])
            s = int((10 * self.population_size) / 100)
            new_generation = [population[tup[0]] for tup in population_fitness[:s]]

            # 90% of the next generation obtained from top 50% of fittest members of current generation
            s = int((90 * self.population_size) / 100)
            for _ in range(s):
                parent1 = random.choice(population[:int(50 * self.population_size / 100)])
                parent2 = random.choice(population[:int(50 * self.population_size / 100)])
                child = self.mate(parent1, parent2)
                new_generation.append(child)

            population = new_generation.copy()

        self.final_cfs = current_best_cf
        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_sparse = copy.deepcopy(self.final_cfs)
            cfs_preds_sparse = copy.deepcopy(self.cfs_preds)
            self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(self.total_CFs, final_cfs_sparse, cfs_preds_sparse,  query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')

        return query_instance