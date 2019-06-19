"""Module to generate diverse counterfactual explanationsbased on tensorflow"""

import tensorflow as tf

import numpy as np
import random
import collections
import timeit

from dice_ml import diverse_counterfactuals as exp


class DiceTensorFlow:

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        # create TensorFLow session if one is not already created
        if tf.get_default_session() is not None:
            self.dice_sess = tf.get_default_session()
        else:
            self.dice_sess = tf.InteractiveSession()

        # initializing model variables
        self.model = model_interface

        # loading trained model
        self.model.load_model()

        # get data-related parameters
        self.data_interface = data_interface
        self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()

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

    def generate_counterfactuals(self, test_instance, total_CFs, desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0, categorical_penalty=1.0, algorithm="DiverseCF", features_to_vary="all", yloss_type="log_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights=None, optimizer="tensorflow:adam", learning_rate=1, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_test_instance=True, tie_random=False, stopping_threshold=0.5):
        """Generates diverse counterfactual explanations

        :param test_instance: Numpy array. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of test_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the test_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.

        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Numpy array of weights for different features in the optimization function. Default option is equal weighting of all features.
        :param optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence
        :param verbose: Print intermediate loss value.
        :param init_near_test_instance: Boolean to indicate if counterfactuals are to be initialized near test_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """

        if not collections.Counter([total_CFs, algorithm, features_to_vary, yloss_type, diversity_loss_type, feature_weights, optimizer]) == collections.Counter(self.cf_init_weights + self.loss_weights + self.optimizer_weights):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
            self.do_optimizer_initializations(optimizer)

        """
        Future Support: We have three main components in our tensorflow graph: (1) initialization of tf.variables (2) defining ops for loss function initializations, and (3) defining ops for optimizer initializations. Define methods to delete some nodes from a tensorflow graphs or update variables/ops in a tensorflow graph dynamically, so that only those components corresponding to the variables that are updated change.
        """

        # check if hyperparameters are to be updated
        if not collections.Counter([proximity_weight, diversity_weight, categorical_penalty]) == collections.Counter(self.hyperparameters):
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        test_instance, test_pred = self.find_counterfactuals(test_instance, desired_class, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_test_instance, tie_random, stopping_threshold)

        return exp.CounterfactualExamples(self.data_interface, test_instance,
        test_pred, self.final_cfs, self.cfs_preds)

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
        return self.dice_sess.run(self.output_tensor, feed_dict={self.input_tensor: input_instance})

    def compute_first_part_of_loss(self, method):
        """Computes the first part (y-loss) of the loss function."""
        loss_part1 = 0.0
        for i in range(self.total_CFs):
            if method == "l2_loss":
                temp_loss = tf.square(tf.subtract(
                    self.model.get_output(self.cfs_frozen[i]), self.target_cf))
            elif method == "log_loss":
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)), tf.subtract(
                    1.0, tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)))))
                temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=temp_logits, labels=self.target_cf)

            loss_part1 = tf.add(loss_part1, temp_loss)

        return tf.divide(loss_part1, tf.to_float(self.total_CFs))

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        if self.feature_weights is not None:
            return tf.reduce_sum(tf.multiply(tf.abs(tf.subtract(x_hat, x1)), self.feature_weights))
        else:
            return tf.reduce_sum(tf.abs(tf.subtract(x_hat, x1)))

    def compute_second_part_of_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        loss_part2 = 0.0
        for i in range(self.total_CFs):
            loss_part2 = tf.add(loss_part2, self.compute_dist(
                self.cfs_frozen[i], self.x1))
        return tf.divide(loss_part2, tf.to_float(tf.multiply(len(self.minx[0]), self.total_CFs)))

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

    def do_loss_initializations(self, yloss_type="l2_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights=None):
        """Defines the optimization loss"""

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        self.loss_weights = [self.yloss_type, self.diversity_loss_type, feature_weights]

        # loss part 1: y-loss
        self.loss_part1 = self.compute_first_part_of_loss(self.yloss_type)

        # loss part 2: similarity between CFs and original instance
        if feature_weights is not None:
            self.feature_weights = tf.Variable(self.minx, dtype=tf.float32)
            self.dice_sess.run(
                tf.assign(self.feature_weights, feature_weights))
        else:
            self.feature_weights = None

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

    def initialize_CFs(self, test_instance, init_near_test_instance=False):
        """Initialize counterfactuals."""
        inits = []
        for n in range(self.total_CFs):
            one_init = []
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_test_instance:
                        one_init.append(test_instance[0][i]+(n*0.01))
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(test_instance[0][i])
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

    def update_hyperparameters(self, proximity_weight=0.5, diversity_weight=0.5, categorical_penalty=1.0):
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
                if self.stopping_threshold < 0.5 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.stopping_threshold > 0.5 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False

    def find_counterfactuals(self, test_instance, desired_class="opposite", learning_rate=1, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_test_instance=True, tie_random=False, stopping_threshold=0.5):
        """Finds counterfactuals by graident-descent."""

        # Prepares user defined test_instance for DiCE.
        test_instance = self.data_interface.prepare_test_instance(test_instance=test_instance, encode=True)
        test_instance = np.array([test_instance.iloc[0].values])

        # find the predicted value of test_instance
        test_pred = self.predict_fn(test_instance)[0][0]
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

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable takes value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1
        for _ in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                init_arrs = self.initialize_CFs(test_instance, False)
            else:
                init_arrs = self.initialize_CFs(test_instance, init_near_test_instance)

            for i in range(0, self.total_CFs):
                self.dice_sess.run(self.cf_assign[i], feed_dict={
                                   self.cf_init: init_arrs[i]})

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # gradient descent step
                _, loss_value = self.dice_sess.run([self.optim_step, self.loss],
                                                   feed_dict={self.learning_rate: learning_rate, self.target_cf: self.target_cf_class, self.x1: test_instance})

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
                    if (iterations+1) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.dice_sess.run(self.cfs[j])
                self.final_cfs.append(temp)

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # calculating separate loss parts
        if self.total_random_inits > 0:
            self.evaluated_loss_parts = self.get_evaluated_loss(test_instance)
        else:
            self.evaluated_loss_parts = []
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part1,
                                                                feed_dict={self.target_cf: self.target_cf_class}))
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part2,
                                                                feed_dict={self.x1: test_instance}))
            self.evaluated_loss_parts.append(
                self.dice_sess.run(self.loss_part3))
            self.evaluated_loss_parts.append(
                self.dice_sess.run(self.loss_part4))

        m, s = divmod(self.elapsed, 60)
        print('Diverse Counterfactuals found! total time taken: %02d' %
              m, 'min %02d' % s, 'sec')

        return test_instance, test_pred

    def get_evaluated_loss(self, test_instance):
        # initiate sess and define TFs
        final_loss = []

        # First part of the loss: y-loss
        if self.yloss_type == "l2_loss":
            loss_part1 = np.sum(np.square(np.subtract(
                self.cfs_preds, self.target_cf_class)))
        elif self.yloss_type == "log_loss":
            loss_part1 = 0.0
            for i in range(len(self.final_cfs)):
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.cfs_preds[i], 0.000001)),
                                               tf.subtract(1.0, tf.abs(tf.subtract(self.cfs_preds[i], 0.000001)))))
                temp_loss = self.dice_sess.run(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=temp_logits, labels=tf.convert_to_tensor(self.target_cf_class, dtype=tf.float32)))
                loss_part1 = np.add(loss_part1, temp_loss)

        # First part of the loss
        final_loss.append(np.divide(loss_part1, len(self.final_cfs)))

        # second part of the loss: dist from x1
        loss_part2 = np.sum(np.absolute(
            np.subtract(self.final_cfs, test_instance)))
        loss_part2 = np.divide(
            loss_part2, (len(self.final_cfs[0][0])*len(self.final_cfs)))
        final_loss.append(loss_part2)

        # third part of the loss: diversity
        if "dpp" in self.diversity_loss_type:
            det_entries = []
            if self.diversity_loss_type.split(':')[1] == "inverse_dist":
                for i in range(len(self.final_cfs)):
                    for j in range(len(self.final_cfs)):
                        det_temp_entry = np.divide(
                            1.0, (1.0 + np.sum(np.absolute(np.subtract(self.final_cfs[i], self.final_cfs[j])))))
                        if i == j:
                            det_temp_entry = det_temp_entry + 0.0001
                        det_entries.append(det_temp_entry)

            elif self.diversity_loss_type.split(':')[1] == "exponential_dist":
                for i in range(len(self.final_cfs)):
                    for j in range(len(self.final_cfs)):
                        det_temp_entry = np.divide(1.0, np.exp(
                            np.sum(np.absolute(np.subtract(self.final_cfs[i], self.final_cfs[j])))))
                        if i == j:
                            det_temp_entry = det_temp_entry + 0.0001
                        det_entries.append(det_temp_entry)

            det_entries = np.reshape(
                det_entries, (len(self.final_cfs), len(self.final_cfs)))
            final_loss.append(np.sum(np.linalg.det(det_entries)))

        elif self.diversity_loss_type == "avg_dist":
            loss_part3 = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(len(self.final_cfs)):
                for j in range(i+1, len(self.final_cfs)):
                    count += 1.0
                    loss_part3 = np.add(loss_part3,
                                        np.divide(1.0, (1.0 + np.sum(np.absolute(np.subtract(self.final_cfs[i], self.final_cfs[j]))))))
            final_loss.append(np.subtract(1.0, np.divide(loss_part3, count)))

        # fourth part: correction loss
        loss_part4 = 0.0
        for i in range(len(self.final_cfs)):
            for v in self.encoded_categorical_feature_indexes:
                loss_part4 = np.add(loss_part4, np.square(np.subtract(
                    np.sum(self.final_cfs[i][0, v[0]:v[-1]+1]), 1.0)))
        final_loss.append(loss_part4)

        return final_loss
