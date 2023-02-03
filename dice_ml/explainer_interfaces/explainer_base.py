"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.explainer_interfaces"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException
from sklearn.neighbors import KDTree
from tqdm import tqdm

from dice_ml.constants import ModelTypes, _PostHocSparsityTypes
from dice_ml.counterfactual_explanations import CounterfactualExplanations


class ExplainerBase(ABC):

    def __init__(self, data_interface, model_interface=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        # initiating data and model related parameters
        self.data_interface = data_interface
        if model_interface is not None:
            # self.data_interface.create_ohe_params()
            self.model = model_interface
            self.model.load_model()  # loading pickled trained model if applicable
            self.model.transformer.feed_data_params(data_interface)
            self.model.transformer.initialize_transform_func()

        # moved the following snippet to a method in public_data_interface
        # self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()
        #
        # # min and max for continuous features in original scale
        # flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        # self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        # org_minx, org_maxx = self.data_interface.get_minx_maxx(normalized=False)
        # self.cont_minx = list(org_minx[0][self.encoded_continuous_feature_indexes])
        # self.cont_maxx = list(org_maxx[0][self.encoded_continuous_feature_indexes])
        #
        # # decimal precisions for continuous features
        # self.cont_precisions = \
        #   [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def _validate_counterfactual_configuration(
            self, query_instances, total_CFs,
            desired_class="opposite", desired_range=None,
            permitted_range=None, features_to_vary="all",
            stopping_threshold=0.5, posthoc_sparsity_param=0.1,
            posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):

        if total_CFs <= 0:
            raise UserConfigValidationException(
                "The number of counterfactuals generated per query instance (total_CFs) should be a positive integer.")

        if features_to_vary != "all":
            if len(features_to_vary) == 0:
                raise UserConfigValidationException("Some features need to be varied for generating counterfactuals.")

        if posthoc_sparsity_algorithm not in _PostHocSparsityTypes.ALL:
            raise UserConfigValidationException(
                'The posthoc_sparsity_algorithm should be {0} and not {1}'.format(
                    ' or '.join(_PostHocSparsityTypes.ALL), posthoc_sparsity_algorithm)
                )

        if stopping_threshold < 0.0 or stopping_threshold > 1.0:
            raise UserConfigValidationException('The stopping_threshold should lie between {0} and {1}'.format(
                str(0.0), str(1.0)))

        if posthoc_sparsity_param is not None and (posthoc_sparsity_param < 0.0 or posthoc_sparsity_param > 1.0):
            raise UserConfigValidationException('The posthoc_sparsity_param should lie between {0} and {1}'.format(
                str(0.0), str(1.0)))

        if self.model is not None and self.model.model_type == ModelTypes.Classifier:
            if desired_range is not None:
                raise UserConfigValidationException(
                    'The desired_range parameter should not be set for classification task')

        if self.model is not None and self.model.model_type == ModelTypes.Regressor:
            if desired_range is None:
                raise UserConfigValidationException(
                    'The desired_range parameter should be set for regression task')

        if desired_range is not None:
            if len(desired_range) != 2:
                raise UserConfigValidationException(
                    "The parameter desired_range needs to have two numbers in ascending order.")
            if desired_range[0] > desired_range[1]:
                raise UserConfigValidationException(
                    "The range provided in desired_range should be in ascending order.")

    def generate_counterfactuals(self, query_instances, total_CFs,
                                 desired_class="opposite", desired_range=None,
                                 permitted_range=None, features_to_vary="all",
                                 stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                 proximity_weight=0.2, sparsity_weight=0.2, diversity_weight=5.0,
                                 categorical_penalty=0.1,
                                 posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):
        """General method for generating counterfactuals.

        :param query_instances: Input point(s) for which counterfactuals are to be generated.
                                This can be a dataframe with one or more rows.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value
                              is "opposite" to the outcome class of query_instance for binary classification.
        :param desired_range: For regression problems. Contains the outcome range to
                              generate counterfactuals in. This should be a list of two numbers in
                              ascending order.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data.
                                If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance. Used by ['genetic', 'gradientdescent'],
                                 ignored by ['random', 'kdtree'] methods.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
                                Used by ['genetic', 'kdtree'], ignored by ['random', 'gradientdescent'] methods.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large (for instance,
                                           income varying from 10k to 1000k) and only if the features share a
                                           monotonic relationship with predicted outcome in the model.
        :param verbose: Whether to output detailed messages.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param kwargs: Other parameters accepted by specific explanation method

        :returns: A CounterfactualExplanations object that contains the list of
                  counterfactual examples per query_instance as one of its attributes.
        """
        self._validate_counterfactual_configuration(
            query_instances=query_instances,
            total_CFs=total_CFs,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range, features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold, posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm, verbose=verbose,
            kwargs=kwargs
        )

        cf_examples_arr = []
        query_instances_list = []
        if isinstance(query_instances, pd.DataFrame):
            for ix in range(query_instances.shape[0]):
                query_instances_list.append(query_instances[ix:(ix+1)])
        elif isinstance(query_instances, Iterable):
            query_instances_list = query_instances
        for query_instance in tqdm(query_instances_list):
            self.data_interface.set_continuous_feature_indexes(query_instance)
            res = self._generate_counterfactuals(
                query_instance, total_CFs,
                desired_class=desired_class,
                desired_range=desired_range,
                permitted_range=permitted_range,
                features_to_vary=features_to_vary,
                stopping_threshold=stopping_threshold,
                posthoc_sparsity_param=posthoc_sparsity_param,
                posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
                verbose=verbose,
                **kwargs)
            cf_examples_arr.append(res)
        self._check_any_counterfactuals_computed(cf_examples_arr=cf_examples_arr)

        return CounterfactualExplanations(cf_examples_list=cf_examples_arr)

    @abstractmethod
    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  permitted_range=None, features_to_vary="all",
                                  stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):
        """Internal method for generating counterfactuals for a given query instance. Any explainerclass
           inherting from this class would need to implement this abstract method.

        :param query_instance: Input point for which counterfactuals are to be generated.
                               This can be a dataframe with one row.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value
                              is "opposite" to the outcome class of query_instance for binary classification.
        :param desired_range: For regression problems. Contains the outcome range to
                              generate counterfactuals in.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data.
                                If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large (for instance,
                                           income varying from 10k to 1000k) and only if the features share a
                                           monotonic relationship with predicted outcome in the model.
        :param verbose: Whether to output detailed messages.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param kwargs: Other parameters accepted by specific explanation method

        :returns: A CounterfactualExplanations object that contains the list of
                  counterfactual examples per query_instance as one of its attributes.
        """
        pass

    def setup(self, features_to_vary, permitted_range, query_instance, feature_weights):
        self.data_interface.check_features_to_vary(features_to_vary=features_to_vary)
        self.data_interface.check_permitted_range(permitted_range)

        if features_to_vary == 'all':
            features_to_vary = self.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.feature_range = self.data_interface.permitted_range
            feature_ranges_orig = self.feature_range
        else:  # compute the new ranges based on user input
            self.feature_range, feature_ranges_orig = self.data_interface.get_features_range(permitted_range)

        self.check_query_instance_validity(features_to_vary, permitted_range, query_instance, feature_ranges_orig)

        return features_to_vary

    def check_query_instance_validity(self, features_to_vary, permitted_range, query_instance, feature_ranges_orig):
        for feature in query_instance:
            if feature == self.data_interface.outcome_name:
                raise ValueError("Target", self.data_interface.outcome_name, "present in query instance")

            if feature not in self.data_interface.feature_names:
                raise ValueError("Feature", feature, "not present in training data!")
        for feature in self.data_interface.categorical_feature_names:
            if query_instance[feature].values[0] not in feature_ranges_orig[feature] and \
                    str(query_instance[feature].values[0]) not in feature_ranges_orig[feature]:
                raise ValueError("Feature", feature, "has a value outside the dataset.")

            if feature not in features_to_vary and permitted_range is not None:
                if feature in permitted_range and feature in self.data_interface.continuous_feature_names:
                    if not permitted_range[feature][0] <= query_instance[feature].values[0] <= permitted_range[feature][1]:
                        raise ValueError("Feature:", feature, "is outside the permitted range and isn't allowed to vary.")
                elif feature in permitted_range and feature in self.data_interface.categorical_feature_names:
                    if query_instance[feature].values[0] not in self.feature_range[feature]:
                        raise ValueError("Feature:", feature, "is outside the permitted range and isn't allowed to vary.")

    def local_feature_importance(self, query_instances, cf_examples_list=None,
                                 total_CFs=10,
                                 desired_class="opposite", desired_range=None, permitted_range=None,
                                 features_to_vary="all", stopping_threshold=0.5,
                                 posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear",
                                 **kwargs):
        """ Estimate local feature importance scores for the given inputs.

        :param query_instances: A list of inputs for which to compute the
                                feature importances. These can be provided as a dataframe.
        :param cf_examples_list: If precomputed, a list of counterfactual
                                 examples for every input point. If cf_examples_list is provided, then
                                 all the following parameters are ignored.
        :param total_CFs: The number of counterfactuals to generate per input
                          (default is 10)
        :param other_parameters: These are the same as the
                                 generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
                  the list of counterfactuals per input, local feature importances per
                  input, and the global feature importance summarized over all inputs.
        """
        self._validate_counterfactual_configuration(
            query_instances=query_instances,
            total_CFs=total_CFs,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range, features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold, posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            kwargs=kwargs
        )
        if cf_examples_list is not None:
            if any([len(cf_examples.final_cfs_df) < 10 for cf_examples in cf_examples_list]):
                raise UserConfigValidationException(
                    "The number of counterfactuals generated per query instance should be "
                    "greater than or equal to 10 to compute feature importance for all query points")
        elif total_CFs < 10:
            raise UserConfigValidationException(
                "The number of counterfactuals requested per "
                "query instance should be greater than or equal to 10 "
                "to compute feature importance for all query points")
        importances = self.feature_importance(
            query_instances,
            cf_examples_list=cf_examples_list,
            total_CFs=total_CFs,
            local_importance=True,
            global_importance=False,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range,
            features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            **kwargs)
        return importances

    def global_feature_importance(self, query_instances, cf_examples_list=None,
                                  total_CFs=10, local_importance=True,
                                  desired_class="opposite", desired_range=None, permitted_range=None,
                                  features_to_vary="all", stopping_threshold=0.5,
                                  posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear",
                                  **kwargs):
        """ Estimate global feature importance scores for the given inputs.

        :param query_instances: A list of inputs for which to compute the
                                feature importances. These can be provided as a dataframe.
        :param cf_examples_list: If precomputed, a list of counterfactual
                                 examples for every input point. If cf_examples_list is provided, then
                                 all the following parameters are ignored.
        :param total_CFs: The number of counterfactuals to generate per input
                          (default is 10)
        :param local_importance: Binary flag indicating whether local feature
                                 importance values should also be returned for each query instance.
        :param other_parameters: These are the same as the generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
                  the list of counterfactuals per input, local feature importances per
                  input, and the global feature importance summarized over all inputs.
        """
        self._validate_counterfactual_configuration(
            query_instances=query_instances,
            total_CFs=total_CFs,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range, features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold, posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            kwargs=kwargs
        )
        if query_instances is not None and len(query_instances) < 10:
            raise UserConfigValidationException(
                "The number of query instances should be greater than or equal to 10 "
                "to compute global feature importance over all query points")
        if cf_examples_list is not None:
            if len(cf_examples_list) < 10:
                raise UserConfigValidationException(
                    "The number of points for which counterfactuals generated should be "
                    "greater than or equal to 10 "
                    "to compute global feature importance")
            elif any([len(cf_examples.final_cfs_df) < 10 for cf_examples in cf_examples_list]):
                raise UserConfigValidationException(
                    "The number of counterfactuals generated per query instance should be "
                    "greater than or equal to 10 "
                    "to compute global feature importance over all query points")
        elif total_CFs < 10:
            raise UserConfigValidationException(
                "The number of counterfactuals requested per query instance should be greater "
                "than or equal to 10 "
                "to compute global feature importance over all query points")
        importances = self.feature_importance(
            query_instances,
            cf_examples_list=cf_examples_list,
            total_CFs=total_CFs,
            local_importance=local_importance,
            global_importance=True,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range,
            features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            **kwargs)
        return importances

    def feature_importance(self, query_instances, cf_examples_list=None,
                           total_CFs=10, local_importance=True, global_importance=True,
                           desired_class="opposite", desired_range=None,
                           permitted_range=None, features_to_vary="all", stopping_threshold=0.5,
                           posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", **kwargs):
        """ Estimate feature importance scores for the given inputs.

        :param query_instances: A list of inputs for which to compute the
                                feature importances. These can be provided as a dataframe.
        :param cf_examples_list: If precomputed, a list of counterfactual
                                 examples for every input point. If cf_examples_list is provided, then
                                 all the following parameters are ignored.
        :param total_CFs: The number of counterfactuals to generate per input
                          (default is 10)
        :param other_parameters: These are the same as the generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
                  the list of counterfactuals per input, local feature importances per
                  input, and the global feature importance summarized over all inputs.
        """
        self._validate_counterfactual_configuration(
            query_instances=query_instances,
            total_CFs=total_CFs,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range, features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold, posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            kwargs=kwargs
        )
        if cf_examples_list is None:
            cf_examples_list = self.generate_counterfactuals(
                query_instances, total_CFs,
                desired_class=desired_class,
                desired_range=desired_range,
                permitted_range=permitted_range,
                features_to_vary=features_to_vary,
                stopping_threshold=stopping_threshold,
                posthoc_sparsity_param=posthoc_sparsity_param,
                posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
                **kwargs).cf_examples_list
        allcols = self.data_interface.categorical_feature_names + self.data_interface.continuous_feature_names
        summary_importance = None
        local_importances = None
        if global_importance:
            summary_importance = {}
            # Initializing importance vector
            for col in allcols:
                summary_importance[col] = 0

        if local_importance:
            local_importances = [{} for _ in range(len(cf_examples_list))]
            # Initializing local importance for the ith query instance
            for i in range(len(cf_examples_list)):
                for col in allcols:
                    local_importances[i][col] = 0

        overall_num_cfs = 0
        # Summarizing the found counterfactuals
        for i in range(len(cf_examples_list)):
            cf_examples = cf_examples_list[i]
            org_instance = cf_examples.test_instance_df

            if cf_examples.final_cfs_df_sparse is not None:
                df = cf_examples.final_cfs_df_sparse
            else:
                df = cf_examples.final_cfs_df

            if df is None:
                continue

            per_query_point_cfs = 0
            for _, row in df.iterrows():
                per_query_point_cfs += 1
                for col in self.data_interface.continuous_feature_names:
                    if not np.isclose(org_instance[col].iat[0], row[col]):
                        if summary_importance is not None:
                            summary_importance[col] += 1
                        if local_importances is not None:
                            local_importances[i][col] += 1
                for col in self.data_interface.categorical_feature_names:
                    if org_instance[col].iat[0] != row[col]:
                        if summary_importance is not None:
                            summary_importance[col] += 1
                        if local_importances is not None:
                            local_importances[i][col] += 1

            if local_importances is not None:
                for col in allcols:
                    if per_query_point_cfs > 0:
                        local_importances[i][col] /= per_query_point_cfs

            overall_num_cfs += per_query_point_cfs

        if summary_importance is not None:
            for col in allcols:
                if overall_num_cfs > 0:
                    summary_importance[col] /= overall_num_cfs

        return CounterfactualExplanations(
            cf_examples_list,
            local_importance=local_importances,
            summary_importance=summary_importance)

    def predict_fn(self, input_instance):
        """prediction function"""

        preds = self.model.get_output(input_instance)
        if self.model.model_type == ModelTypes.Classifier and \
           len(preds.shape) == 1:  # from deep learning predictors
            preds = np.column_stack([1 - preds, preds])
        return preds

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        return self.model.get_output(input_instance)

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, query_instance, posthoc_sparsity_param,
                                        posthoc_sparsity_algorithm, limit_steps_ls):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: Final CFs in original user-fed format, in a pandas dataframe.
        :param query_instance: Query instance in original user-fed format, in a pandas dataframe.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search.
                                           Prefer binary search when a feature range is
                                           large (for instance, income varying from 10k to 1000k)
                                           and only if the features share a monotonic relationship
                                           with predicted outcome in the model.
        :param limit_steps_ls: Defines the limit of steps to be done in the linear search,
                                necessary to avoid infinite loops
        """
        if final_cfs_sparse is None:
            return final_cfs_sparse

        # quantiles of the deviation from median for every continuous feature
        quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param)
        mads = self.data_interface.get_valid_mads()
        # Setting the quantile of a feature to be the minimum of mad and quantile
        # Thus, the maximum deviation can be mad.
        for feature in quantiles:
            quantiles[feature] = min(quantiles[feature], mads[feature])

        # Sorting features such that the feature with the highest quantile deviation
        # is first
        features_sorted = sorted(quantiles.items(), key=lambda kv: kv[1], reverse=True)
        for ix in range(len(features_sorted)):
            features_sorted[ix] = features_sorted[ix][0]
        precs = self.data_interface.get_decimal_precisions()
        decimal_prec = dict(zip(self.data_interface.continuous_feature_names, precs))

        cfs_preds_sparse = []

        for cf_ix in list(final_cfs_sparse.index):
            current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
            for feature in features_sorted:
                # current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iat[[cf_ix]][self.data_interface.feature_names])
                # feat_ix = self.data_interface.continuous_feature_names.index(feature)
                diff = query_instance[feature].iat[0] - final_cfs_sparse.at[cf_ix, feature]
                if(abs(diff) <= quantiles[feature]):
                    if posthoc_sparsity_algorithm == "linear":
                        final_cfs_sparse = self.do_linear_search(diff, decimal_prec, query_instance, cf_ix,
                                                                 feature, final_cfs_sparse, current_pred, limit_steps_ls)

                    elif posthoc_sparsity_algorithm == "binary":
                        final_cfs_sparse = self.do_binary_search(
                            diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

            temp_preds = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
            cfs_preds_sparse.append(temp_preds[0])
        final_cfs_sparse[self.data_interface.outcome_name] = self.get_model_output_from_scores(cfs_preds_sparse)
        # final_cfs_sparse[self.data_interface.outcome_name] = np.round(final_cfs_sparse[self.data_interface.outcome_name], 3)
        return final_cfs_sparse

    def do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse,
                         current_pred_orig, limit_steps_ls):
        """Performs a greedy linear search - moves the continuous features in CFs towards original values in
           query_instance greedily until the prediction class changes, or it reaches the maximum number of steps"""

        old_diff = diff
        change = (10**-decimal_prec[feature])  # the minimal possible change for a feature
        current_pred = current_pred_orig
        count_steps = 0
        if self.model.model_type == ModelTypes.Classifier:
            while((abs(diff) > 10e-4) and (np.sign(diff*old_diff) > 0) and
                  self.is_cf_valid(current_pred)) and (count_steps < limit_steps_ls):

                old_val = final_cfs_sparse.at[cf_ix, feature]
                final_cfs_sparse.at[cf_ix, feature] += np.sign(diff)*change
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])
                old_diff = diff

                if not self.is_cf_valid(current_pred):
                    final_cfs_sparse.at[cf_ix, feature] = old_val
                    return final_cfs_sparse

                diff = query_instance[feature].iat[0] - final_cfs_sparse.at[cf_ix, feature]

                count_steps += 1

        return final_cfs_sparse

    def do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred):
        """Performs a binary search between continuous features of a CF and corresponding values
           in query_instance until the prediction class changes."""

        old_val = final_cfs_sparse.at[cf_ix, feature]
        final_cfs_sparse.at[cf_ix, feature] = query_instance[feature].iat[0]
        # Prediction of the query instance
        current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

        # first check if assigning query_instance values to a CF is required.
        if self.is_cf_valid(current_pred):
            return final_cfs_sparse
        else:
            final_cfs_sparse.at[cf_ix, feature] = old_val

        # move the CF values towards the query_instance
        if diff > 0:
            left = final_cfs_sparse.at[cf_ix, feature]
            right = query_instance[feature].iat[0]

            while left <= right:
                current_val = left + ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.at[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.is_cf_valid(current_pred):
                    left = current_val + (10 ** -decimal_prec[feature])
                else:
                    right = current_val - (10 ** -decimal_prec[feature])

        else:
            left = query_instance[feature].iat[0]
            right = final_cfs_sparse.at[cf_ix, feature]

            while right >= left:
                current_val = right - ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.at[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.loc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.is_cf_valid(current_pred):
                    right = current_val - (10**-decimal_prec[feature])
                else:
                    left = current_val + (10**-decimal_prec[feature])

        return final_cfs_sparse

    def misc_init(self, stopping_threshold, desired_class, desired_range, test_pred):
        self.stopping_threshold = stopping_threshold
        if self.model.model_type == ModelTypes.Classifier:
            self.target_cf_class = np.array(
                [[self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)]],
                dtype=np.float32)
            desired_class = int(self.target_cf_class[0][0])
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        elif self.model.model_type == ModelTypes.Regressor:
            self.target_cf_range = self.infer_target_cfs_range(desired_range)
        return desired_class

    def infer_target_cfs_class(self, desired_class_input, original_pred, num_output_nodes):
        """ Infer the target class for generating CFs. Only called when
            model_type=="classifier".
            TODO: Add support for opposite desired class in multiclass.
            Downstream methods should decide whether it is allowed or not.
        """
        if desired_class_input == "opposite":
            if num_output_nodes == 2:
                # 'original_pred' needs to be converted to the proper class if 'original_pred' comes from the
                # 'predict_proba' method (length is 2 and class is index with maximum value). Otherwise 'original_pred'
                # already contains the class.
                if hasattr(original_pred, "__len__") and len(original_pred) > 1:
                    original_pred_1 = np.argmax(original_pred)
                else:
                    original_pred_1 = original_pred
                target_class = int(1 - original_pred_1)
                return target_class
            elif num_output_nodes == 1:  # only for pytorch DL model
                original_pred_1 = np.round(original_pred)
                target_class = int(1-original_pred_1)
                return target_class
            elif num_output_nodes > 2:
                raise UserConfigValidationException(
                    "Desired class cannot be opposite if the number of classes is more than 2.")
        elif isinstance(desired_class_input, int):
            if num_output_nodes == 1:   # for DL models
                if desired_class_input in (0, 1):
                    target_class = desired_class_input
                    return target_class
                else:
                    raise UserConfigValidationException("Only 0, 1 are supported as desired class for binary classification!")
            elif desired_class_input >= 0 and desired_class_input < num_output_nodes:
                target_class = desired_class_input
                return target_class
            else:
                raise UserConfigValidationException("Desired class not present in training data!")
        else:
            raise UserConfigValidationException("The target class for {0} could not be identified".format(
                                                desired_class_input))

    def infer_target_cfs_range(self, desired_range_input):
        target_range = None
        if desired_range_input is None:
            raise ValueError("Need to provide a desired_range for the target counterfactuals for a regression model.")
        else:
            if desired_range_input[0] > desired_range_input[1]:
                raise ValueError("Invalid Range!")
            else:
                target_range = desired_range_input
        return target_range

    def decide_cf_validity(self, model_outputs):
        validity = np.zeros(len(model_outputs), dtype=np.int32)
        for i in range(len(model_outputs)):
            pred = model_outputs[i]
            if self.model.model_type == ModelTypes.Classifier:
                if self.num_output_nodes in (1, 2):  # binary
                    if self.num_output_nodes == 2:
                        pred_1 = pred[self.num_output_nodes-1]
                    else:
                        pred_1 = pred[0]
                    validity[i] = 1 if \
                        ((self.target_cf_class == 0 and pred_1 <= self.stopping_threshold) or
                         (self.target_cf_class == 1 and pred_1 >= self.stopping_threshold)) else 0

                else:  # multiclass
                    if np.argmax(pred) == self.target_cf_class:
                        validity[i] = 1
            elif self.model.model_type == ModelTypes.Regressor:
                if self.target_cf_range[0] <= pred <= self.target_cf_range[1]:
                    validity[i] = 1
        return validity

    def is_cf_valid(self, model_score):
        """Check if a cf belongs to the target class or target range.
        """
        # Converting to single prediction if the prediction is provided as a
        # singleton array
        correct_dim = 1 if self.model.model_type == ModelTypes.Classifier else 0
        if hasattr(model_score, "shape") and len(model_score.shape) > correct_dim:
            model_score = model_score[0]
        # Converting target_cf_class to a scalar (tf/torch have it as (1,1) shape)
        if self.model.model_type == ModelTypes.Classifier:
            target_cf_class = self.target_cf_class
            if hasattr(self.target_cf_class, "shape"):
                if len(self.target_cf_class.shape) == 1:
                    target_cf_class = self.target_cf_class[0]
                elif len(self.target_cf_class.shape) == 2:
                    target_cf_class = self.target_cf_class[0][0]
            target_cf_class = int(target_cf_class)

            if len(model_score) == 1:  # for tensorflow/pytorch models
                pred_1 = model_score[0]
                validity = True if \
                    ((target_cf_class == 0 and pred_1 <= self.stopping_threshold) or
                     (target_cf_class == 1 and pred_1 >= self.stopping_threshold)) else False
                return validity
            elif len(model_score) == 2:  # binary
                pred_1 = model_score[1]
                validity = True if \
                    ((target_cf_class == 0 and pred_1 <= self.stopping_threshold) or
                     (target_cf_class == 1 and pred_1 >= self.stopping_threshold)) else False
                return validity
            else:  # multiclass
                return np.argmax(model_score) == target_cf_class
        else:
            return self.target_cf_range[0] <= model_score and model_score <= self.target_cf_range[1]

    def get_model_output_from_scores(self, model_scores):
        if self.model.model_type == ModelTypes.Classifier:
            output_type = np.int32
        else:
            output_type = np.float32
        model_output = np.zeros(len(model_scores), dtype=output_type)
        for i in range(len(model_scores)):
            if self.model.model_type == ModelTypes.Classifier:
                if hasattr(model_scores[i], "shape") and len(model_scores[i].shape) > 0:
                    if model_scores[i].shape[0] > 1:
                        model_output[i] = np.argmax(model_scores[i])
                    else:
                        model_output[i] = np.round(model_scores[i])[0]
                else:  # 1-D input
                    model_output[i] = np.round(model_scores[i])
            elif self.model.model_type == ModelTypes.Regressor:
                model_output[i] = model_scores[i]
        return model_output

    def check_permitted_range(self, permitted_range):
        """checks permitted range for continuous features
           TODO: add comments as to where this is used if this function is necessary, else remove.
        """
        if permitted_range is not None:
            # if not self.data_interface.check_features_range(permitted_range):
            #   raise ValueError(
            #       "permitted range of features should be within their original range")
            # else:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

    def sigmoid(self, z):
        """This is used in VAE-based CF explainers."""
        return 1 / (1 + np.exp(-z))

    def build_KD_tree(self, data_df_copy, desired_range, desired_class, predicted_outcome_name):
        # Stores the predictions on the training data
        dataset_instance = self.data_interface.prepare_query_instance(
            query_instance=data_df_copy[self.data_interface.feature_names])

        predictions = self.model.get_output(dataset_instance, model_score=False).flatten()
        # TODO: Is it okay to insert a column in the original dataframe with the predicted outcome? This is memory-efficient
        data_df_copy[predicted_outcome_name] = predictions

        # segmenting the dataset according to outcome
        dataset_with_predictions = None
        if self.model.model_type == ModelTypes.Classifier:
            dataset_with_predictions = data_df_copy.loc[[i == desired_class for i in predictions]].copy()

        elif self.model.model_type == ModelTypes.Regressor:
            dataset_with_predictions = data_df_copy.loc[
                [desired_range[0] <= pred <= desired_range[1] for pred in predictions]].copy()

        KD_tree = None
        # Prepares the KD trees for DiCE
        if len(dataset_with_predictions) > 0:
            dummies = pd.get_dummies(dataset_with_predictions[self.data_interface.feature_names])
            KD_tree = KDTree(dummies)

        return dataset_with_predictions, KD_tree, predictions

    def round_to_precision(self):
        # to display the values with the same precision as the original data
        precisions = self.data_interface.get_decimal_precisions()
        for ix, feature in enumerate(self.data_interface.continuous_feature_names):
            self.final_cfs_df[feature] = self.final_cfs_df[feature].astype(float).round(precisions[ix])
            if self.final_cfs_df_sparse is not None:
                self.final_cfs_df_sparse[feature] = self.final_cfs_df_sparse[feature].astype(float).round(precisions[ix])

    def _check_any_counterfactuals_computed(self, cf_examples_arr):
        """Check if any counterfactuals were generated for any query point."""
        no_cf_generated = True
        # Check if any counterfactuals were generated for any query point
        for cf_examples in cf_examples_arr:
            if cf_examples.final_cfs_df is not None and len(cf_examples.final_cfs_df) > 0:
                no_cf_generated = False
                break
        if no_cf_generated:
            raise UserConfigValidationException(
                "No counterfactuals found for any of the query points! Kindly check your configuration.")

    def serialize_explainer(self, path):
        """Serialize the explainer to the file specified by path."""
        with open(path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def deserialize_explainer(path):
        """Reload the explainer into the memory by reading the file specified by path."""
        deserialized_exp = None
        with open(path, "rb") as pickle_file:
            deserialized_exp = pickle.load(pickle_file)

        return deserialized_exp
