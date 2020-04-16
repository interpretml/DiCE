"""
Module to generate diverse counterfactual explanations based on tensorflow 1.x
"""
from dice_ml.dice_interfaces.dice_base import DiceBase
import tensorflow as tf

import numpy as np
import random
import collections
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp

class DiceTensorFlow1(DiceBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface) # initiating data related parameters

        # create TensorFLow session if one is not already created
        if tf.get_default_session() is not None:
            self.dice_sess = tf.get_default_session()
        else:
            self.dice_sess = tf.InteractiveSession()

        # initializing model variables
        self.model = model_interface

        # loading trained model
        self.model.load_model()

        self.input_tensor = tf.Variable(self.minx, dtype=tf.float32)
        self.output_tensor = self.model.get_output(self.input_tensor)

        # hyperparameter initializations
        self.weights = []
        self.weights_inits = tf.placeholder(tf.float32, shape=())
        self.weights_assign = []
        for i in range(3):
            self.weights.append(tf.Variable(1.0, dtype=tf.float32))
            self.weights_assign.append(tf.assign(self.weights[i], self.weights_inits))

        self.hyperparameters = []  # proximity_weight, diversity_weight, categorical_penalty
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.optimizer_weights = []  # optimizer

        # number of output nodes of ML model
        temp_input = tf.convert_to_tensor([tf.random.uniform([len(self.data_interface.encoded_feature_names)])], dtype=tf.float32)
        self.num_ouput_nodes = self.dice_sess.run(self.model.get_output(temp_input)).shape[1]

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, stopping_threshold=0.5, posthoc_sparsity_param=0.1):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.

        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
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

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """

        if([total_CFs, algorithm, features_to_vary, yloss_type, diversity_loss_type, feature_weights, optimizer] != (self.cf_init_weights + self.loss_weights + self.optimizer_weights)):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
            self.do_optimizer_initializations(optimizer)

        """
        Future Support: We have three main components in our tensorflow graph: (1) initialization of tf.variables (2) defining ops for loss function initializations, and (3) defining ops for optimizer initializations. Need to define methods to delete some nodes from a tensorflow graphs or update variables/ops in a tensorflow graph dynamically, so that only those components corresponding to the variables that are updated change.
        """

        # check if hyperparameters are to be updated
        if not collections.Counter([proximity_weight, diversity_weight, categorical_penalty]) == collections.Counter(self.hyperparameters):
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        query_instance, test_pred = self.find_counterfactuals(query_instance, desired_class, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param)

        return exp.CounterfactualExamples(self.data_interface, query_instance,
        test_pred, self.final_cfs, self.cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param)

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes TF variables required for CF generation."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # a placeholder for original instance
        self.x1 = tf.placeholder(tf.float32, shape=(1, self.minx.shape[1]))

        # target CF placeholder
        self.target_cf = tf.placeholder(tf.float32, (1, 1))

        # learning rate for GD
        self.learning_rate = tf.placeholder(tf.float32, ())

        # CF initializations
        self.cfs = []
        self.cf_init = tf.placeholder(
            tf.float32, shape=(1, self.minx.shape[1]))
        self.cf_assign = []
        for i in range(self.total_CFs):
            self.cfs.append(tf.Variable(self.minx, dtype=tf.float32))
            self.cf_assign.append(tf.assign(self.cfs[i], self.cf_init))

        # freezing those columns that need to be fixed
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)

        frozen_dims = np.array([[1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(self.minx.shape[1])]])
        self.cfs_frozen = []
        for i in range(self.total_CFs):
            frozen = tf.multiply(self.cfs[i], tf.constant(
                frozen_dims, dtype=tf.float32))
            self.cfs_frozen.append(
                frozen + tf.stop_gradient(-frozen + self.cfs[i]))

        # Future Support: a dictionary of options for optimizers, only checked with tensorflow optimizers: need to check with others
        self.optimizers_options = {
            "tensorflow": self.tensorflow_optimizers,
            "scipy": self.scipy_optimizers
        }

    def predict_fn(self, input_instance):
        """prediction function"""
        temp_preds = self.dice_sess.run(self.output_tensor, feed_dict={self.input_tensor: input_instance})
        return np.array([preds[(self.num_ouput_nodes-1):] for preds in temp_preds])

    def compute_first_part_of_loss(self, method):
        """Computes the first part (y-loss) of the loss function."""
        loss_part1 = 0.0
        for i in range(self.total_CFs):
            if method == "l2_loss":
                temp_loss = tf.square(tf.subtract(
                    self.model.get_output(self.cfs_frozen[i]), self.target_cf))
                temp_loss = temp_loss[:,(self.num_ouput_nodes-1):]
            elif method == "log_loss":
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)), tf.subtract(
                    1.0, tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)))))
                temp_logits = temp_logits[:,(self.num_ouput_nodes-1):]
                temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=temp_logits, labels=self.target_cf)
            elif method == "hinge_loss":
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)), tf.subtract(
                    1.0, tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)))))
                temp_logits = temp_logits[:,(self.num_ouput_nodes-1):]
                temp_loss = tf.losses.hinge_loss(
                    logits=temp_logits, labels=self.target_cf)

            loss_part1 = tf.add(loss_part1, temp_loss)

        return tf.divide(loss_part1, tf.cast(self.total_CFs, dtype=tf.float32))

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply(tf.abs(tf.subtract(x_hat, x1)), self.feature_weights))

    def compute_second_part_of_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        loss_part2 = 0.0
        for i in range(self.total_CFs):
            loss_part2 = tf.add(loss_part2, self.compute_dist(
                self.cfs_frozen[i], self.x1))
        return tf.divide(loss_part2, tf.cast(tf.multiply(len(self.minx[0]), self.total_CFs), dtype=tf.float32))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = []
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.add(
                        1.0, self.compute_dist(self.cfs_frozen[i], self.cfs_frozen[j])))
                    if i == j:
                        det_temp_entry = tf.add(det_temp_entry, 0.0001)
                    det_entries.append(det_temp_entry)

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.exp(
                        self.compute_dist(self.cfs_frozen[i], self.cfs_frozen[j])))
                    det_entries.append(det_temp_entry)

        det_entries = tf.reshape(det_entries, [self.total_CFs, self.total_CFs])
        loss_part3 = tf.matrix_determinant(det_entries)
        return loss_part3

    def compute_third_part_of_loss(self, method):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return tf.constant(0.0)

        if "dpp" in method:
            submethod = method.split(':')[1]
            return tf.reduce_sum(self.dpp_style(submethod))
        elif method == "avg_dist":
            loss_part3 = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    loss_part3 = tf.add(loss_part3,
                                        tf.divide(1.0, tf.add(1.0, self.compute_dist(self.cfs_frozen[i], self.cfs_frozen[j]))))
            return tf.subtract(1.0, tf.divide(loss_part3, count))

    def compute_fourth_part_of_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
        loss_part4 = tf.constant(0.0)
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                loss_part4 = tf.add(loss_part4, tf.square(tf.subtract(
                    tf.reduce_sum(self.cfs_frozen[i][0, v[0]:v[-1]+1]), 1.0)))
        return loss_part4

    def do_loss_initializations(self, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad"):
        """Defines the optimization loss"""

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        self.loss_weights = [self.yloss_type, self.diversity_loss_type, feature_weights]

        # loss part 1: y-loss
        self.loss_part1 = self.compute_first_part_of_loss(self.yloss_type)

        # loss part 2: similarity between CFs and original instance
        if feature_weights == "inverse_mad":
            normalized_mads = self.data_interface.get_mads(normalized=True)
            feature_weights = {}
            for feature in normalized_mads:
                feature_weights[feature] = round(1/normalized_mads[feature], 2)

        feature_weights_list = []
        for feature in self.data_interface.encoded_feature_names:
            if feature in feature_weights:
                feature_weights_list.append(feature_weights[feature])
            else:
                feature_weights_list.append(1.0)
        feature_weights_list = [feature_weights_list]

        self.feature_weights = tf.Variable(self.minx, dtype=tf.float32)
        self.dice_sess.run(
            tf.assign(self.feature_weights, np.array(feature_weights_list, dtype=np.float32)))

        self.loss_part2 = self.compute_second_part_of_loss()

        # loss part 3: diversity between CFs
        if self.total_random_inits > 0:
            # random initialization method
            self.loss_part3 = tf.constant(0.0, dtype=tf.float32)
        else:
            self.loss_part3 = self.compute_third_part_of_loss(self.diversity_loss_type)

        # loss part 4: diversity between CFs
        self.loss_part4 = self.compute_fourth_part_of_loss()

        # final loss:
        self.loss = tf.add(tf.subtract(tf.add(self.loss_part1, tf.scalar_mul(self.weights[0], self.loss_part2)), tf.scalar_mul(self.weights[1], self.loss_part3)), tf.scalar_mul(self.weights[2], self.loss_part4))

    def tensorflow_optimizers(self, method="adam"):
        """Initializes tensorflow optimizers."""
        if method == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate, name='myadam')
            #opt = tf.contrib.optimizer_v2.AdamOptimizer(self.learning_rate)
        elif method == "rmsprop":
            opt = tf.train.RMSPropOptimizer(self.learning_rate)

        optim_step = opt.minimize(self.loss, var_list=self.cfs)
        return opt, optim_step

    # Future Support: implement scipt wrappers
    def scipy_optimizers(self, method="Nelder-Mead"):
        opt = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss, var_list=self.cfs, method='Nelder-Mead')
        optim_step = opt.minimize(self.loss, var_list=self.cfs)
        return optim_step

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        inits = []
        for n in range(self.total_CFs):
            one_init = []
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        one_init.append(query_instance[0][i]+(n*0.01))
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(query_instance[0][i])
            inits.append(np.array([one_init]))
        return inits

    def do_optimizer_initializations(self, optimizer):
        """Initializes gradient-based TF optimizers."""

        self.optimizer_weights = [optimizer]

        opt_library = optimizer.split(':')[0]
        opt_method = optimizer.split(':')[1]

        # optimization step
        self.opt, self.optim_step = self.optimizers_options[opt_library](opt_method)

        self.opt_vars = self.opt.variables()
        self.reset_optimizer_op = tf.variables_initializer(self.opt_vars)
        self.dice_sess.run(self.reset_optimizer_op)

    def update_hyperparameters(self, proximity_weight=0.5, diversity_weight=0.5, categorical_penalty=0.1):
        """Updates hyperparameters."""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.dice_sess.run(self.weights_assign[0], feed_dict={self.weights_inits: proximity_weight})
        if self.total_random_inits > 0:
            self.dice_sess.run(self.weights_assign[1], feed_dict={
                               self.weights_inits: 0.0})
        else:
            self.dice_sess.run(self.weights_assign[1], feed_dict={
                               self.weights_inits: diversity_weight})
        # just sanity check. anyways, the loss part is made to 0.
        self.dice_sess.run(self.weights_assign[2], feed_dict={self.weights_inits: categorical_penalty})

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = self.dice_sess.run(tcf)
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                org_cont = (cf[0, v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i] # continuous feature in orginal scale
                org_cont = round(org_cont, self.cont_precisions[i]) # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[0, v] = normalized_cont # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[0, v[0]:v[-1]+1] == np.amax(cf[0, v[0]:v[-1]+1])).flatten().tolist()
                if(len(maxs) > 1):
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[0, v[vi]] = 1.0
                    else:
                        cf[0, v[vi]] = 0.0
            temp_cfs.append(cf)
            if assign:
                self.dice_sess.run(self.cf_assign[index], feed_dict={
                                   self.cf_init: cf})
        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if((self.project_iter > 0) & (itr > 0)):
            if((itr % self.project_iter) == 0):
                self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                temp_cfs = self.round_off_cfs(assign=False)
                cfs_preds = [self.predict_fn(cf) for cf in temp_cfs]
                test_preds = [np.round(preds.flatten().tolist(), 3)
                              for preds in cfs_preds]
                test_preds = [
                    item for sublist in test_preds for item in sublist]
                if self.target_cf_class[0][0] == 0 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.target_cf_class[0][0] == 1 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False

    def find_counterfactuals(self, query_instance, desired_class="opposite", learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=False, tie_random=False, stopping_threshold=0.5, posthoc_sparsity_param=0.1):
        """Finds counterfactuals by graident-descent."""

        # Prepares user defined query_instance for DiCE.
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values], dtype=np.float32)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)[0][0]
        if desired_class == "opposite":
            desired_class = 1.0 - round(test_pred)
        self.target_cf_class = np.array([[desired_class]])

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class[0][0] == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class[0][0] == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        self.dice_sess.run(self.reset_optimizer_op)

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # variables to backup best known CFs so far in the optimization process - if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = []
        self.best_backup_cfs_preds = []
        self.min_dist_from_threshold = 100

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1
        for _ in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                init_arrs = self.initialize_CFs(query_instance, False)
            else:
                init_arrs = self.initialize_CFs(query_instance, init_near_query_instance)

            for i in range(0, self.total_CFs):
                self.dice_sess.run(self.cf_assign[i], feed_dict={
                                   self.cf_init: init_arrs[i]})

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # gradient descent step
                _, loss_value = self.dice_sess.run([self.optim_step, self.loss],
                                                   feed_dict={self.learning_rate: learning_rate, self.target_cf: self.target_cf_class, self.x1: query_instance})

                # projection step
                for j in range(0, self.total_CFs):
                    temp_cf = self.dice_sess.run(self.cfs[j])
                    clip_cf = np.clip(temp_cf, self.minx, self.maxx)  # clipping
                    # to remove -ve sign before 0.0 in some cases
                    clip_cf = np.add(clip_cf, np.array(
                        [np.zeros([self.minx.shape[1]])]))
                    self.dice_sess.run(self.cf_assign[j], feed_dict={
                                       self.cf_init: clip_cf})

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

                if((self.target_cf_class[0][0] == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) | (self.target_cf_class[0][0] == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean([abs(pred-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold:
                        self.min_dist_from_threshold = avg_preds_dist
                        self.best_backup_cfs = temp_cfs_stored
                        self.best_backup_cfs_preds = test_preds_stored

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.dice_sess.run(self.cfs[j])
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        self.valid_cfs_found = False
        if((self.target_cf_class[0][0] == 0 and any(i > self.stopping_threshold for i in test_preds_stored)) | (self.target_cf_class[0][0] == 1 and any(i < self.stopping_threshold for i in test_preds_stored))):
            if self.min_dist_from_threshold != 100:
                for ix in range(self.total_CFs):
                    self.final_cfs[ix] = self.best_backup_cfs[ix]
                    self.cfs_preds[ix] = self.best_backup_cfs_preds[ix]

                self.valid_cfs_found = True # final_cfs have valid CFs through backup CFs
            else:
                self.valid_cfs_found = False # neither final_cfs nor backup cfs are valid
        else:
            self.valid_cfs_found = True # final_cfs have valid CFs

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            self.final_cfs_sparse = copy.deepcopy(self.final_cfs)
            self.cfs_preds_sparse = copy.deepcopy(self.cfs_preds)

            normalized_quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param, normalized=True)
            normalized_mads = self.data_interface.get_mads(normalized=True)
            for feature in normalized_quantiles:
                normalized_quantiles[feature] = min(normalized_quantiles[feature], normalized_mads[feature])

            features_sorted = sorted(normalized_quantiles.items(), key=lambda kv: kv[1], reverse=True)
            for ix in range(len(features_sorted)):
                features_sorted[ix] = features_sorted[ix][0]
            decimal_prec = self.data_interface.get_decimal_precisions()[0:len(self.encoded_continuous_feature_indexes)]

            for cf_ix in range(self.total_CFs):
                for feature in features_sorted:
                    current_pred = self.predict_fn(self.final_cfs_sparse[cf_ix])
                    feat_ix = self.data_interface.encoded_feature_names.index(feature)
                    change = (10**-decimal_prec[feat_ix])/(self.cont_maxx[feat_ix] - self.cont_minx[feat_ix])
                    diff = query_instance[0][feat_ix] - self.final_cfs_sparse[cf_ix][0][feat_ix]
                    old_diff = diff

                    if(abs(diff) <= normalized_quantiles[feature]):
                        while((abs(diff)>10e-4) & (np.sign(diff*old_diff) > 0) &
                              ((self.target_cf_class[0][0] == 0 and current_pred[0][0] < self.stopping_threshold) |
                               (self.target_cf_class[0][0] == 1 and current_pred[0][0] > self.stopping_threshold))):
                            old_val = self.final_cfs_sparse[cf_ix][0][feat_ix]
                            self.final_cfs_sparse[cf_ix][0][feat_ix] += np.sign(diff)*change
                            current_pred = self.predict_fn(self.final_cfs_sparse[cf_ix])
                            old_diff = diff

                            if(((self.target_cf_class[0][0] == 0 and current_pred[0][0] > self.stopping_threshold) | (self.target_cf_class[0][0] == 1 and current_pred[0][0] < self.stopping_threshold))):
                                self.final_cfs_sparse[cf_ix][0][feat_ix] = old_val
                                diff = query_instance[0][feat_ix] - self.final_cfs_sparse[cf_ix][0][feat_ix]
                                break

                            diff = query_instance[0][feat_ix] - self.final_cfs_sparse[cf_ix][0][feat_ix]

                self.cfs_preds_sparse[cf_ix] = self.predict_fn(self.final_cfs_sparse[cf_ix])
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            self.total_CFs_found = self.total_CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            for pred in self.cfs_preds:
                if((self.target_cf_class[0][0] == 0 and pred[0][0] < self.stopping_threshold) | (self.target_cf_class[0][0] == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1

            print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...' % (self.total_CFs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return query_instance, test_pred
