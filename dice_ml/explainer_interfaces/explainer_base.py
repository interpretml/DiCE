"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.explainer_interfaces"""

import numpy as np

class ExplainerBase:

    def __init__(self, data_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        """
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

    def generate_counterfactuals(self):
        raise NotImplementedError

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
