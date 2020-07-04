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

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: list of final CFs in numpy format.
        :param cfs_preds_sparse: list of predicted outcomes of final CFs in numpy format.
        :param query_instance: query instance in numpy format.
        :param posthoc_sparsity_param: parameter for the post-hoc operation on continuous features to enhance sparsity.
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
                change = (10**-decimal_prec[feat_ix])/(self.cont_maxx[feat_ix] - self.cont_minx[feat_ix])
                diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]
                old_diff = diff

                if(abs(diff) <= normalized_quantiles[feature]):
                    while((abs(diff)>10e-4) and (np.sign(diff*old_diff) > 0) and
                          ((self.target_cf_class == 0 and current_pred < self.stopping_threshold) |
                           (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                        old_val = final_cfs_sparse[cf_ix].ravel()[feat_ix]
                        final_cfs_sparse[cf_ix].ravel()[feat_ix] += np.sign(diff)*change
                        current_pred = self.predict_fn(final_cfs_sparse[cf_ix])
                        old_diff = diff

                        if(((self.target_cf_class == 0 and current_pred > self.stopping_threshold) | (self.target_cf_class == 1 and current_pred < self.stopping_threshold))):
                            final_cfs_sparse[cf_ix].ravel()[feat_ix] = old_val
                            diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]
                            break

                        diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]

            cfs_preds_sparse[cf_ix] = self.predict_fn(final_cfs_sparse[cf_ix])

        return final_cfs_sparse, cfs_preds_sparse
