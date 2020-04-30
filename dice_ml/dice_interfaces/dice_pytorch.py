"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
from dice_ml.dice_interfaces.dice_base import DiceBase
import torch

import numpy as np
import random
import collections
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp

class DicePyTorch(DiceBase):

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

        # set the model in evaluation mode
        ev = self.model.set_eval_mode()

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate

        # number of output nodes of ML model
        temp_input = torch.rand([len(self.data_interface.encoded_feature_names)]).float()
        self.num_ouput_nodes = len(self.model.get_output(temp_input).data)

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all", yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500, max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, stopping_threshold=0.5, posthoc_sparsity_param=0.1):
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
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".

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

        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if([yloss_type, diversity_loss_type, feature_weights] != self.loss_weights):
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if([proximity_weight, diversity_weight, categorical_penalty] != self.hyperparameters):
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        query_instance, test_pred = self.find_counterfactuals(query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param)

        return exp.CounterfactualExamples(self.data_interface, query_instance,
        test_pred, self.final_cfs, self.cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param)

    def get_model_output(self, input_instance):
        """get output probability of ML model"""
        return self.model.get_output(input_instance)[(self.num_ouput_nodes-1):]

    def predict_fn(self, input_instance):
        """prediction function"""
        return self.get_model_output(input_instance).data

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

        # CF initialization
        if len(self.cfs) == 0:
            for ix in range(self.total_CFs):
                one_init = []
                for jx in range(self.minx.shape[1]):
                    one_init.append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(torch.tensor(one_init).float())
                self.cfs[ix].requires_grad = True

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
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
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = torch.tensor(feature_weights_list)

        # define different parts of loss function
        self.yloss_opt = torch.nn.BCEWithLogitsLoss()

    def update_hyperparameters(self, proximity_weight, diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based PyTorch optimizers."""
        opt_library = optimizer.split(':')[0]
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        for i in range(self.total_CFs):
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow((self.get_model_output(self.cfs[i]) - self.target_cf_class), 2)[0]
            elif self.yloss_type == "log_loss":
                temp_logits = torch.log10((abs(self.get_model_output(self.cfs[i]) - 0.000001))/(1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.BCEWithLogitsLoss()
                temp_loss = criterion(temp_logits, torch.tensor([self.target_cf_class]))
            elif self.yloss_type == "hinge_loss":
                temp_logits = torch.log10((abs(self.get_model_output(self.cfs[i]) - 0.000001))/(1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.ReLU()
                temp_loss = criterion(0.5 - (temp_logits*self.target_cf_class))[0]

            yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i,j)] += 0.0001

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i,j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += torch.pow((torch.sum(self.cfs[i][v[0]:v[-1]+1]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss()
        self.diversity_loss = self.compute_diversity_loss()
        self.regularization_loss = self.compute_regularization_loss()

        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - (self.diversity_weight * self.diversity_loss) + (self.categorical_penalty * self.regularization_loss)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cfs[n].data[i] = query_instance[i]+(n*0.01)
                    else:
                        self.cfs[n].data[i] = np.random.uniform(self.minx[0][i], self.maxx[0][i])
                else:
                    self.cfs[n].data[i] = query_instance[i]

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.detach().clone().numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                org_cont = (cf[v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i] # continuous feature in orginal scale
                org_cont = round(org_cont, self.cont_precisions[i]) # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[v] = normalized_cont # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if(len(maxs) > 1):
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0

            temp_cfs.append(torch.tensor(cf))
            if assign:
                for jx in range(len(cf)):
                    self.cfs[index].data[jx] = temp_cfs[index][jx]

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
                test_preds = [self.predict_fn(cf)[0] for cf in temp_cfs]

                if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param):
        """Finds counterfactuals by graident-descent."""

        # Prepares user defined query_instance for DiCE.
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = query_instance.iloc[0].values
        self.x1 = torch.tensor(query_instance)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(torch.tensor(query_instance).float())
        test_pred = test_pred.numpy()[0]
        if desired_class == "opposite":
            desired_class = 1.0 - round(test_pred)
        self.target_cf_class = torch.tensor(desired_class).float()

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

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
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropogate
                loss_value = self.compute_loss()
                self.loss.backward()

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                # update the variables
                self.optimizer.step()

                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

                if((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) | (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean([abs(pred[0]-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold:
                        self.min_dist_from_threshold = avg_preds_dist
                        self.best_backup_cfs = temp_cfs_stored
                        self.best_backup_cfs_preds = test_preds_stored

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].detach().clone()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        self.valid_cfs_found = False
        if((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in test_preds_stored)) | (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in test_preds_stored))):
            if self.min_dist_from_threshold != 100:
                for ix in range(self.total_CFs):
                    self.final_cfs[ix] = self.best_backup_cfs[ix].clone()
                    self.cfs_preds[ix] = self.best_backup_cfs_preds[ix].clone()

                self.valid_cfs_found = True # final_cfs have valid CFs through backup CFs
            else:
                self.valid_cfs_found = False # neither final_cfs nor backup cfs are valid
        else:
            self.valid_cfs_found = True # final_cfs have valid CFs

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            self.final_cfs_sparse = []
            self.cfs_preds_sparse = []
            for ix in range(self.total_CFs):
                self.final_cfs_sparse.append(self.final_cfs[ix].clone())
                self.cfs_preds_sparse.append(self.cfs_preds[ix].clone())

            normalized_quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param, normalized=True)
            normalized_mads = self.data_interface.get_valid_mads(normalized=True)
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
                    diff = query_instance[feat_ix] - self.final_cfs_sparse[cf_ix][feat_ix]
                    old_diff = diff

                    if(abs(diff) <= normalized_quantiles[feature]):
                        while((abs(diff)>10e-4) & (np.sign(diff*old_diff) > 0) &
                              ((self.target_cf_class == 0 and current_pred[0] < self.stopping_threshold) |
                               (self.target_cf_class == 1 and current_pred[0] > self.stopping_threshold))):
                            old_val = self.final_cfs_sparse[cf_ix][feat_ix]
                            self.final_cfs_sparse[cf_ix][feat_ix] += np.sign(diff)*change
                            current_pred = self.predict_fn(self.final_cfs_sparse[cf_ix])
                            old_diff = diff

                            if(((self.target_cf_class == 0 and current_pred[0] > self.stopping_threshold) | (self.target_cf_class == 1 and current_pred[0] < self.stopping_threshold))):
                                self.final_cfs_sparse[cf_ix][feat_ix] = old_val
                                diff = query_instance[feat_ix] - self.final_cfs_sparse[cf_ix][feat_ix]
                                break

                            diff = query_instance[feat_ix] - self.final_cfs_sparse[cf_ix][feat_ix]

                self.cfs_preds_sparse[cf_ix] = self.predict_fn(self.final_cfs_sparse[cf_ix])
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        # convert to the format that is consistent with dice_tensorflow
        for tix in range(self.total_CFs):
            temp = self.final_cfs[tix].clone().numpy()
            self.final_cfs[tix] = np.array([temp], dtype=np.float32)

            temp = self.cfs_preds[tix].clone().numpy()
            self.cfs_preds[tix] = np.array([temp], dtype=np.float32)

            if self.final_cfs_sparse is not None:
                temp = self.final_cfs_sparse[tix].clone().numpy()
                self.final_cfs_sparse[tix] = np.array([temp], dtype=np.float32)

                temp = self.cfs_preds_sparse[tix].clone().numpy()
                self.cfs_preds_sparse[tix] = np.array([temp], dtype=np.float32)

            if len(self.best_backup_cfs) > 0:
                temp = self.best_backup_cfs[tix].clone().numpy()
                self.best_backup_cfs[tix] = np.array([temp], dtype=np.float32)

                temp = self.best_backup_cfs_preds[tix].clone().numpy()
                self.best_backup_cfs_preds[tix] = np.array([temp], dtype=np.float32)

        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            self.total_CFs_found = self.total_CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            for pred in self.cfs_preds:
                if((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1

            print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...' % (self.total_CFs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return query_instance, test_pred
