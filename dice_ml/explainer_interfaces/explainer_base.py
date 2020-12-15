"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.explainer_interfaces"""

import numpy as np
import pandas as pd
import random
import timeit
import copy

from dice_ml import diverse_counterfactuals as exp

class ExplainerBase:

    def __init__(self, data_interface, model_interface=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        self.model = model_interface
        # get data-related parameters - minx and max for normalized continuous features
        self.data_interface = data_interface
        self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()

        # min and max for continuous features in original scale
        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        org_minx, org_maxx = self.data_interface.get_minx_maxx(normalized=False)
        self.cont_minx = list(org_minx[0][self.encoded_continuous_feature_indexes])
        self.cont_maxx = list(org_maxx[0][self.encoded_continuous_feature_indexes])

        # decimal precisions for continuous features
        self.cont_precisions = [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", permitted_range=None, features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", sample_size=1000, random_seed=17):
        """Generate counterfactuals by randomly sampling features.

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        """

        # permitted range for continuous features
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

        # fixing features that are to be fixed
        self.total_CFs = total_CFs
        if features_to_vary == "all":
            self.fixed_features_values = {}
        else:
            self.fixed_features_values = {}
            for feature in self.data_interface.feature_names:
                if feature not in features_to_vary:
                    self.fixed_features_values[feature] = query_instance[feature]

        # number of output nodes of ML model
        temp_input = np.random.rand(1,len(self.data_interface.encoded_feature_names))
        self.num_ouput_nodes = len(self.model.get_output(temp_input))

        # Prepares user defined query_instance for DiCE.
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        query_instance = np.array([query_instance.iloc[0].values], dtype=np.float32)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)[0]
        if desired_class == "opposite":
            desired_class = 1.0 - round(test_pred)
        self.target_cf_class = desired_class

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # get random samples for each feature independently
        start_time = timeit.default_timer()
        samples = get_samples(self, self.fixed_features_values, sampling_random_seed=random_seed, sampling_size=sample_size)

        cfs = self.data_interface.prepare_query_instance(query_instance=samples, encode=True).values
        cf_preds = self.predict_fn(cfs)
        cfs_df = pd.DataFrame(np.append(cfs, np.array([cf_preds]).T, axis=1), columns = self.data_interface.encoded_feature_names + [self.data_interface.outcome_name])

        # check validity of CFs
        cfs_df['validity'] = cfs_df[self.data_interface.outcome_name].apply(lambda pred: 1 if ((self.target_cf_class == 0 and pred<= self.stopping_threshold) or (self.target_cf_class == 1 and pred>= self.stopping_threshold)) else 0)
        self.total_CFs_found = cfs_df[cfs_df['validity']==1].shape[0]

        if self.total_CFs_found >= self.total_CFs:
            cfs_df = cfs_df[cfs_df['validity']==1].sample(n=self.total_CFs, random_state=random_seed)
            self.valid_cfs_found = True
        else:
            temp_df = cfs_df[cfs_df['validity']==0].sample(n=self.total_CFs-self.total_CFs_found, random_state=random_seed)
            cfs_df = pd.concat([cfs_df[cfs_df['validity']==1], temp_df], ignore_index=True)
            self.valid_cfs_found = False

        # convert to the format that is consistent with dice_tensorflow
        temp = cfs_df[self.data_interface.encoded_feature_names].values
        self.final_cfs = [np.array([arr]) for arr in temp]
        temp = cfs_df[[self.data_interface.outcome_name]].values
        self.cfs_preds = [np.array([arr]) for arr in temp]

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_sparse = copy.deepcopy(self.final_cfs)
            cfs_preds_sparse = copy.deepcopy(self.cfs_preds)
            self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_sparse, cfs_preds_sparse,  query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            self.final_cfs_sparse = None
            self.cfs_preds_sparse = None

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...' % (self.total_CFs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return exp.CounterfactualExamples(self.data_interface, query_instance,
        test_pred, self.final_cfs, self.cfs_preds, self.final_cfs_sparse, self.cfs_preds_sparse, posthoc_sparsity_param, desired_class)


    def predict_fn(self, input_instance):
        """prediction function"""
        return self.model.get_output(input_instance)[:,(self.num_ouput_nodes-1)]

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: List of final CFs in numpy format.
        :param cfs_preds_sparse: List of predicted outcomes of final CFs in numpy format.
        :param query_instance: Query instance in numpy format.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        """

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
                current_pred = self.predict_fn(final_cfs_sparse[cf_ix])
                feat_ix = self.data_interface.encoded_feature_names.index(feature)
                diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]

                if(abs(diff) <= normalized_quantiles[feature]):
                    if posthoc_sparsity_algorithm == "linear":
                        final_cfs_sparse[cf_ix] = do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred)

                    elif posthoc_sparsity_algorithm == "binary":
                        final_cfs_sparse[cf_ix] = do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred)

            cfs_preds_sparse[cf_ix] = self.predict_fn(final_cfs_sparse[cf_ix])

        return final_cfs_sparse, cfs_preds_sparse

def get_samples(self, fixed_features_values, sampling_random_seed, sampling_size):

    # first get required parameters
    precisions = self.data_interface.get_decimal_precisions()[0:len(self.encoded_continuous_feature_indexes)]

    categorical_features_frequencies = {}
    for feature in self.data_interface.categorical_feature_names:
        categorical_features_frequencies[feature] = len(self.data_interface.train_df[feature].value_counts())

    if sampling_random_seed is not None:
        random.seed(sampling_random_seed)

    samples = []
    for feature in self.data_interface.feature_names:
        if feature in fixed_features_values:
            sample = [fixed_features_values[feature]]*sampling_size
        elif feature in self.data_interface.continuous_feature_names:
            low, high = self.data_interface.permitted_range[feature]
            feat_ix = self.data_interface.encoded_feature_names.index(feature)
            sample = get_continuous_samples(low, high, precisions[feat_ix], size=sampling_size, seed=sampling_random_seed)
        else:
            if sampling_random_seed is not None:
                random.seed(sampling_random_seed)
            sample = random.choices(self.data_interface.train_df[feature].unique(), k=sampling_size)

        samples.append(sample)

    samples = pd.DataFrame(dict(zip(self.data_interface.feature_names, samples))).to_dict(orient='records')#.values
    return samples


def get_continuous_samples(low, high, precision, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if precision == 0:
        result = np.random.randint(low, high+1, size).tolist()
        result = [float(r) for r in result]
    else:
        result = np.random.uniform(low, high+(10**-precision), size)
        result = [round(r, precision) for r in result]
    return result

def do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred):
    """Performs a greedy linear search - moves the continuous features in CFs towards original values in query_instance greedily until the prediction class changes."""

    old_diff = diff
    change = (10**-decimal_prec[feat_ix])/(self.cont_maxx[feat_ix] - self.cont_minx[feat_ix]) # the minimal possible change for a feature
    while((abs(diff)>10e-4) and (np.sign(diff*old_diff) > 0) and
          ((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or
           (self.target_cf_class == 1 and current_pred > self.stopping_threshold))): # move until the prediction class changes
        old_val = final_cfs_sparse[cf_ix].ravel()[feat_ix]
        final_cfs_sparse[cf_ix].ravel()[feat_ix] += np.sign(diff)*change
        current_pred = self.predict_fn(final_cfs_sparse[cf_ix])
        old_diff = diff

        if(((self.target_cf_class == 0 and current_pred > self.stopping_threshold) or (self.target_cf_class == 1 and current_pred < self.stopping_threshold))):
            final_cfs_sparse[cf_ix].ravel()[feat_ix] = old_val
            diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]
            return final_cfs_sparse[cf_ix]

        diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]

    return final_cfs_sparse[cf_ix]

def do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred):
    """Performs a binary search between continuous features of a CF and corresponding values in query_instance until the prediction class changes."""

    old_val = final_cfs_sparse[cf_ix].ravel()[feat_ix]
    final_cfs_sparse[cf_ix].ravel()[feat_ix] = query_instance.ravel()[feat_ix]
    current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

    # first check if assigning query_instance values to a CF is required.
    if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
        return final_cfs_sparse[cf_ix]
    else:
        final_cfs_sparse[cf_ix].ravel()[feat_ix] = old_val

    # move the CF values towards the query_instance
    if diff > 0:
        left = final_cfs_sparse[cf_ix].ravel()[feat_ix]
        right = query_instance.ravel()[feat_ix]

        while left <= right:
            current_val = left + ((right - left)/2)
            current_val = round(current_val, decimal_prec[feat_ix])

            final_cfs_sparse[cf_ix].ravel()[feat_ix] = current_val
            current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

            if current_val == right or current_val == left:
                break

            if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                left = current_val + (10**-decimal_prec[feat_ix])
            else:
                right = current_val - (10**-decimal_prec[feat_ix])

    else:
        left = query_instance.ravel()[feat_ix]
        right = final_cfs_sparse[cf_ix].ravel()[feat_ix]

        while right >= left:
            current_val = right - ((right - left)/2)
            current_val = round(current_val, decimal_prec[feat_ix])

            final_cfs_sparse[cf_ix].ravel()[feat_ix] = current_val
            current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

            if current_val == right or current_val == left:
                break

            if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                right = current_val - (10**-decimal_prec[feat_ix])
            else:
                left = current_val + (10**-decimal_prec[feat_ix])

    return final_cfs_sparse[cf_ix]
