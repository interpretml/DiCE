"""Module to generate diverse counterfactual explanations based on tensorflow framework."""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import random
import pandas as pd
import collections

import timeit


from dice_ml import diverse_counterfactuals as exp

class DiceTensorFlow:

    def __init__(self, data_interface, model_interface):
        """Init method

        Args:
            data_interface: an interface class to access data related params.
            model_interface: an interface class to access trained ML model.
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
        self.minx, self.maxx, self.cont_vars_idx, self.enc_cat_feat_ind = self.data_interface.get_data_params()

        self.input_tensor = tf.Variable(self.minx, dtype=tf.float32)
        self.output_tensor = self.model.get_output(self.input_tensor)

        # hyperparameter initializations
        self.lams = []
        self.lam_init = tf.placeholder(tf.float32, shape=())
        self.lam_assign = []
        for i in range(3):
            self.lams.append(tf.Variable(1.0, dtype=tf.float32))
            self.lam_assign.append(tf.assign(self.lams[i], self.lam_init))

        self.hyperparameters = [] # lam1, lam2, lam3
        self.cfinit_params = [] # total_CFs, algorithm, features_to_vary
        self.loss_params = [] # first_part, third_part, feature_weights
        self.optimizer_params = [] # optimizer

    def generate_counterfactuals(self, input, total_CFs, desired_class="opposite", lam1=0.5, lam2=1.0, lam3=1.0, algorithm="DiverseCF", features_to_vary="all", first_part="log_loss", third_part="dpp_style:inverse_dist", feature_weights=None, optimizer="tensorflow:adam", learning_rate=1, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_x1=True, tie_random=False, stopping_threshold=0.5):
        """Generates diverse counterfactual explanations

        Args:
            input: Numpy array. Test point of interest.
            total_CFs: Total number of counterfactuals required.

            desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of input for binary classification.
            lam1: A positive float. Larger this hyperparameter value, more close the counterfactuals are to the test input.
            lam2: A positive float. Larger this hyperparameter value, more diverse the counterfactuals are.
            lam3: A positive float. Hyperparameter to ensure that all levels of a categorical variable sums to 1.

            algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
            features_to_vary: Either a string "all" or a list of feature names to vary.
            first_part: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss".
            thrid_part: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
            feature_weights: Numpy array of weights for different features in the optimization function. Default option is equal weighting of all features.
            optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

            learning_rate: Learning rate for optimizer.
            min_iter: Min iterations to run gradient descent for.
            max_iter: Max iterations to run gradient descent for.
            project_iter: Project the gradients at an interval of these many iterations.
            loss_diff_thres: Minimum difference between successive loss values to check convergence.
            loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence
            verbose: Print intermediate loss value.
            init_near_x1: Boolean to indicate if counterfactuals are to be initialized near test_input.
            tie_random: Used in rounding off CFs and intermediate projection.
            stopping_threshold: Minimum threshold for counterfactuals target class probability.

        Returns:
            A DiverseCounterfactuals object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).
        """

        if not collections.Counter([total_CFs, algorithm, features_to_vary, first_part, third_part, feature_weights, optimizer]) == collections.Counter(self.cfinit_params + self.loss_params + self.optimizer_params):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
            self.do_loss_initializations(first_part, third_part, feature_weights)
            self.do_optimizer_initializations(optimizer)

        """
        Note: We have three main components in our tensorflow graph: (1) initialization of tf.variables (2) defining ops for loss function initializations, and (3) defining ops for optimizer initializations.
        TODO: Delete some nodes from a tensorflow graphs or update variables or ops in a tensorflow graph dynamically so that only those components corresponding to the variables that are updated change.
        """

        #check if hyperparameters are to be updated
        if not collections.Counter([lam1, lam2, lam3]) == collections.Counter(self.hyperparameters):
            self.update_hyperparameters(lam1, lam2, lam3)

        test_instance, test_pred = self.find_counterfactuals(input, desired_class, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_x1, tie_random, stopping_threshold)

        return exp.DiverseCounterfactuals(self.data_interface, test_instance, test_pred, self.final_cfs, self.cfs_preds)

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes TF variables required for CF generation."""

        self.cfinit_params = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            self.total_random_inits = total_CFs # no. of times to run the experiment with random inits for diversity
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs # size of counterfactual set

        # a placeholder for original instance
        self.x1 = tf.placeholder(tf.float32, shape=(1, self.minx.shape[1]))

        # target CF placeholder
        self.target_cf = tf.placeholder(tf.float32, (1,1))

        # a placeholder for original instance
        self.x1 = tf.placeholder(tf.float32, shape=(1, self.minx.shape[1]))

        # target CF placeholder
        self.target_cf = tf.placeholder(tf.float32, (1,1))

        # learning rate for GD
        self.learning_rate = tf.placeholder(tf.float32, ())

        # CF initializations
        self.cfs = []
        self.cf_init = tf.placeholder(tf.float32, shape=(1, self.minx.shape[1]))
        self.cf_assign = []
        for i in range(self.total_CFs):
            self.cfs.append(tf.Variable(self.minx, dtype=tf.float32))
            self.cf_assign.append(tf.assign(self.cfs[i], self.cf_init))

        # freezing those columns that need to be fixed
            self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)

        frozen_dims = np.array([[1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(self.minx.shape[1])]])
        self.cfs_frozen = []
        for i in range(self.total_CFs):
            frozen = tf.multiply(self.cfs[i], tf.constant(frozen_dims, dtype=tf.float32))
            self.cfs_frozen.append(frozen + tf.stop_gradient(-frozen + self.cfs[i]))

        # a dictionary of options for optimizers
        ## only checked with tensorflow optimizers. ##TODO: need to check with others
        self.options_dict = {
            "tensorflow": self.tensorflow_optimizers,
            "scipy": self.scipy_optimizers
        }


    def predict_fn(self, input_instance):
        """prediction function"""
        return self.dice_sess.run(self.output_tensor, feed_dict={self.input_tensor:input_instance})

    def compute_first_part_of_loss(self, method):
        """Computes the first part (y-loss) of the loss function."""
        loss_part1 = 0.0
        for i in range(self.total_CFs):
            if method == "l2_loss":
                temp_loss = tf.square(tf.subtract(self.model.get_output(self.cfs_frozen[i]), self.target_cf))
            elif method == "log_loss":
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)), tf.subtract(1.0, tf.abs(tf.subtract(self.model.get_output(self.cfs_frozen[i]), 0.000001)))))
                temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=temp_logits, labels=self.target_cf)

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
            loss_part2 = tf.add(loss_part2, self.compute_dist(self.cfs_frozen[i], self.x1))
        return tf.divide(loss_part2, tf.to_float(tf.multiply(len(self.minx[0]), self.total_CFs)))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = []
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.add(1.0, self.compute_dist(self.cfs_frozen[i], self.cfs_frozen[j])))
                    if i==j:
                        det_temp_entry = tf.add(det_temp_entry, 0.0001)
                    det_entries.append(det_temp_entry)

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.exp(self.compute_dist(self.cfs_frozen[i], self.cfs_frozen[j])))
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
            for v in self.enc_cat_feat_ind:
                loss_part4 = tf.add(loss_part4, tf.square(tf.subtract(tf.reduce_sum(self.cfs_frozen[i][0,v[0]:v[-1]+1]), 1.0)))
        #return tf.subtract(loss_part4, tf.to_float(tf.multiply(self.total_CFs, len(self.enc_cat_feat_ind))))
        return loss_part4

    def do_loss_initializations(self, first_part="l2_loss", third_part="dpp_style:inverse_dist", feature_weights=None):
        """Defines the optimization loss"""

        # define the loss parts
        self.first_part = first_part
        self.third_part = third_part

        self.loss_params = [self.first_part, self.third_part, feature_weights]

        # loss part 1: y-loss
        self.loss_part1 = self.compute_first_part_of_loss(self.first_part)

        # loss part 2: similarity between CFs and original instance
        if feature_weights is not None:
           self.feature_weights = tf.Variable(self.minx, dtype=tf.float32)
           self.dice_sess.run(tf.assign(self.feature_weights, feature_weights))
        else:
           self.feature_weights = None

        self.loss_part2 = self.compute_second_part_of_loss()

        # loss part 3: diversity between CFs
        if self.total_random_inits > 0: self.loss_part3 = tf.constant(0.0, dtype=tf.float32)  # random initialization method
        else: self.loss_part3 = self.compute_third_part_of_loss(self.third_part)

        # loss part 4: diversity between CFs
        self.loss_part4 = self.compute_fourth_part_of_loss()

        # final loss:
        #self.loss = self.loss_part1 + tf.scalar_mul(self.lam1, self.loss_part2)- tf.scalar_mul(self.lam2, self.loss_part3)
        self.loss = tf.add(tf.subtract(tf.add(self.loss_part1, tf.scalar_mul(self.lams[0], self.loss_part2)),
                                    tf.scalar_mul(self.lams[1], self.loss_part3)),
                                    tf.scalar_mul(self.lams[2], self.loss_part4))

    def tensorflow_optimizers(self, method="adam"):
        """Initializes tensorflow optimizers."""
        if method == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate, name='myadam')
            #opt = tf.contrib.optimizer_v2.AdamOptimizer(self.learning_rate)
        elif method == "rmsprop":
            opt = tf.train.RMSPropOptimizer(self.learning_rate)

        optim_step = opt.minimize(self.loss, var_list = self.cfs)
        return opt, optim_step

    # TODO: implement scipt wrappers
    def scipy_optimizers(self, method="Nelder-Mead"):
        opt = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list = self.cfs, method='Nelder-Mead')
        optim_step = opt.minimize(self.loss, var_list = self.cfs)
        return optim_step

    def initialize_CFs(self, test_input, init_near_x1=False):
        """Initialize counterfactuals."""
        inits = []
        #rand = 9  ## for reproducibility
        for n in range(self.total_CFs):
            one_init = []
            #np.random.seed(rand + n)
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_x1:
                        one_init.append(test_input[0][i]+(n*0.01))
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(test_input[0][i])
            inits.append(np.array([one_init]))
        return inits

    def do_optimizer_initializations(self, optimizer):
        """Initializes gradient-based TF optimizers."""

        self.optimizer_params = [optimizer]

        opt_library = optimizer.split(':')[0]
        opt_method = optimizer.split(':')[1]

        # optimization step
        self.opt, self.optim_step = self.options_dict[opt_library](opt_method)

        self.opt_vars = self.opt.variables()
        self.reset_optimizer_op = tf.variables_initializer(self.opt_vars)
        self.dice_sess.run(self.reset_optimizer_op)

    def update_hyperparameters(self, lam1=0.5, lam2=0.5, lam3=1.0):
        """Updates hyperparameters."""

        self.hyperparameters = [lam1, lam2, lam3]
        self.dice_sess.run(self.lam_assign[0], feed_dict={self.lam_init: lam1})
        if self.total_random_inits > 0: self.dice_sess.run(self.lam_assign[1], feed_dict={self.lam_init: 0.0})
        else: self.dice_sess.run(self.lam_assign[1], feed_dict={self.lam_init: lam2})
        self.dice_sess.run(self.lam_assign[2], feed_dict={self.lam_init: lam3}) # just sanity check. anyways, the loss part is made to 0.

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = self.dice_sess.run(tcf)
            for v in self.enc_cat_feat_ind:
                maxs = np.argwhere(cf[0,v[0]:v[-1]+1] == np.amax(cf[0,v[0]:v[-1]+1])).flatten().tolist()
                if(len(maxs) > 1):
                    if self.tie_random: ix = random.choice(maxs)
                    else: ix = maxs[0]
                else: ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix: cf[0, v[vi]] = 1.0
                    else: cf[0, v[vi]] = 0.0
            temp_cfs.append(cf)
            if assign:
                self.dice_sess.run(self.cf_assign[index], feed_dict={self.cf_init: cf})
        if assign: return None
        else: return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if((self.project_iter>0)&(itr>0)):
            if((itr%self.project_iter)==0): self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr<self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr>=self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter: return False
            else:
                temp_cfs = self.round_off_cfs(assign=False)
                cfs_preds = [self.predict_fn(cf) for cf in temp_cfs]
                test_preds = [np.round(preds.flatten().tolist(), 3) for preds in cfs_preds]
                test_preds = [item for sublist in test_preds for item in sublist]
                if self.stopping_threshold<0.5 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.stopping_threshold>0.5 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else: return False ## should add self.loss_converge_iter=0 here too? TODO:
        else:
            self.loss_converge_iter = 0
            return False

    def find_counterfactuals(self, input, desired_class="opposite", learning_rate=1, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_x1=True, tie_random=False, stopping_threshold=0.5):
        """Finds counterfactuals by graident-descent."""

        # Prepares user defined test input for DiCE.
        test_input = self.data_interface.get_test_inputs(params=input, encode=True)
        test_input = np.array([test_input.iloc[0].values])

        # find the predicted value of test_input
        test_pred = self.predict_fn(test_input)[0][0]
        if desired_class == "opposite":
            desired_class = 1.0 - round(test_pred)
        self.target_cf_class = np.array([[desired_class]])

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        self.loss_converge_maxiter = loss_converge_maxiter # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_iter = 0
        self.converged = False


        self.stopping_threshold = stopping_threshold
        if self.target_cf_class[0][0]==0 and self.stopping_threshold>0.5: self.stopping_threshold = 0.25
        elif self.target_cf_class[0][0]==1 and self.stopping_threshold<0.5: self.stopping_threshold = 0.75

        self.dice_sess.run(self.reset_optimizer_op)

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable takes value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        #looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1
        for _ in range(loop_find_CFs):
            # CF init
            if self.total_random_inits>0: init_arrs = self.initialize_CFs(test_input, False)
            else: init_arrs = self.initialize_CFs(test_input, init_near_x1)

            for i in range(0, self.total_CFs):
                self.dice_sess.run(self.cf_assign[i], feed_dict={self.cf_init: init_arrs[i]})

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # gradient descent step
                _, loss_value = self.dice_sess.run([self.optim_step, self.loss],
                                         feed_dict={self.learning_rate: learning_rate, self.target_cf: self.target_cf_class, self.x1: test_input})

                # projection step
                for j in range(0, self.total_CFs):
                    temp_cf = self.dice_sess.run(self.cfs[j])
                    clip_cf = np.clip(temp_cf, self.minx, self.maxx) # clipping
                    clip_cf = np.add(clip_cf, np.array([np.zeros([self.minx.shape[1]])]))  # to remove -ve sign before 0.0 in some cases
                    self.dice_sess.run(self.cf_assign[j], feed_dict={self.cf_init: clip_cf})

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
            self.evaluated_loss_parts = self.get_evaluated_loss(test_input)
        else:
            self.evaluated_loss_parts = []
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part1,
                                                                feed_dict={self.target_cf: self.target_cf_class}))
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part2,
                                                                feed_dict={self.x1: test_input}))
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part3))
            self.evaluated_loss_parts.append(self.dice_sess.run(self.loss_part4))

        m, s = divmod(self.elapsed, 60)
        print('Diverse Counterfactuals found! total time taken: %02d' %m, 'min %02d' %s, 'sec')

        return test_input, test_pred

    def get_evaluated_loss(self, test_input):
        # initiate sess and define TFs
        final_loss = []

        # First part of the loss: y-loss
        if self.first_part == "l2_loss":
            loss_part1 = np.sum(np.square(np.subtract(self.cfs_preds, self.target_cf_class)))
        elif self.first_part == "log_loss":
            loss_part1 = 0.0
            for i in range(len(self.final_cfs)):
                temp_logits = tf.log(tf.divide(tf.abs(tf.subtract(self.cfs_preds[i], 0.000001)),
                                                   tf.subtract(1.0, tf.abs(tf.subtract(self.cfs_preds[i], 0.000001)))))
                temp_loss = self.dice_sess.run(tf.nn.sigmoid_cross_entropy_with_logits(logits=temp_logits, labels=tf.convert_to_tensor(self.target_cf_class, dtype=tf.float32)))
                loss_part1 = np.add(loss_part1, temp_loss)

        final_loss.append(np.divide(loss_part1, len(self.final_cfs))) #First part of the loss

        # second part of the loss: dist from x1
        loss_part2 = np.sum(np.absolute(np.subtract(self.final_cfs, test_input)))
        loss_part2 = np.divide(loss_part2, (len(self.final_cfs[0][0])*len(self.final_cfs)))
        final_loss.append(loss_part2)

        # third part of the loss: diversity
        if "dpp" in self.third_part:
            det_entries = []
            if self.third_part.split(':')[1] == "inverse_dist":
                for i in range(len(self.final_cfs)):
                    for j in range(len(self.final_cfs)):
                        det_temp_entry = np.divide(1.0, (1.0 + np.sum(np.absolute(np.subtract(self.final_cfs[i], self.final_cfs[j])))))
                        if i==j:
                            det_temp_entry = det_temp_entry + 0.0001
                        det_entries.append(det_temp_entry)

            elif self.third_part.split(':')[1] == "exponential_dist":
                for i in range(len(self.final_cfs)):
                    for j in range(len(self.final_cfs)):
                        det_temp_entry = np.divide(1.0, np.exp(np.sum(np.absolute(np.subtract(self.final_cfs[i], self.final_cfs[j])))))
                        if i==j:
                            det_temp_entry = det_temp_entry + 0.0001
                        det_entries.append(det_temp_entry)

            det_entries = np.reshape(det_entries, (len(self.final_cfs), len(self.final_cfs)))
            final_loss.append(np.sum(np.linalg.det(det_entries)))

        elif self.third_part == "avg_dist":
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
            for v in self.enc_cat_feat_ind:
                loss_part4 = np.add(loss_part4, np.square(np.subtract(np.sum(self.final_cfs[i][0,v[0]:v[-1]+1]), 1.0)))
        final_loss.append(loss_part4)

        return final_loss
