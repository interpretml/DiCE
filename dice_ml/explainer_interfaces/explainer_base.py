"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.explainer_interfaces"""

import numpy as np
import pandas as pd
import random
import timeit
import copy
from collections.abc import Iterable
from sklearn.neighbors import KDTree

import dice_ml.diverse_counterfactuals as exp
from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.utils.exception import UserConfigValidationException


class ExplainerBase:

    def __init__(self, data_interface, model_interface=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """

        # initiating data and model related parameters
        self.data_interface = data_interface
        if model_interface is not None:
            #self.data_interface.create_ohe_params()
            self.model = model_interface
            self.model.load_model() # loading pickled trained model if applicable
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
                # self.cont_precisions = [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def generate_counterfactuals(self, query_instances, total_CFs,
            desired_class="opposite", desired_range=None,
            permitted_range=None, features_to_vary="all",
            stopping_threshold=0.5, posthoc_sparsity_param=0.1,
            posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):
        """Generate counterfactuals by randomly sampling features.

        :param query_instances: Input point(s) for which counterfactuals are to be generated. This can be a dataframe with one or more rows.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data. If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        :param verbose: Whether to output detailed messages.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param kwargs: Other parameters accepted by specific explanation method

        :returns: A CounterfactualExplanations object that contains the list of
        counterfactual examples per query_instance as one of its attributes.
        """

        cf_examples_arr = []
        query_instances_list = []
        if isinstance(query_instances, pd.DataFrame):
            for ix in range(query_instances.shape[0]):
                query_instances_list.append(query_instances[ix:(ix+1)])
        elif isinstance(query_instances, Iterable):
            query_instances_list = query_instances
            #query_instances = query_instances.to_dict("records")
        for query_instance in query_instances_list:
            res = self._generate_counterfactuals(query_instance, total_CFs,
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
        return CounterfactualExplanations(cf_examples_list=cf_examples_arr)

    def setup(self, features_to_vary, permitted_range, query_instance, feature_weights):
        if features_to_vary == 'all':
            features_to_vary = self.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.feature_range = self.data_interface.permitted_range
            feature_ranges_orig = self.feature_range
        else: # compute the new ranges based on user input
            self.feature_range, feature_ranges_orig = self.data_interface.get_features_range(permitted_range)
        self.check_query_instance_validity(features_to_vary, permitted_range, query_instance, feature_ranges_orig)

        # check feature MAD validity and throw warnings
        self.check_mad_validity(feature_weights)

        return features_to_vary

    def check_query_instance_validity(self, features_to_vary, permitted_range, query_instance, feature_ranges_orig):
        for feature in self.data_interface.categorical_feature_names:
            if query_instance[feature].values[0] not in feature_ranges_orig[feature]:
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
        if cf_examples_list is not None:
            if any([len(cf_examples.final_cfs_df) < 10 for cf_examples in cf_examples_list]):
                raise UserConfigValidationException("The number of counterfactuals generated per query instance should be greater than or equal to 10")
        elif total_CFs < 10:
            raise UserConfigValidationException("The number of counterfactuals generated per query instance should be greater than or equal to 10")
        importances = self.feature_importance(query_instances,
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
        :param other_parameters: These are the same as the
        generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
        the list of counterfactuals per input, local feature importances per
        input, and the global feature importance summarized over all inputs.
        """
        if len(query_instances) < 10:
            raise UserConfigValidationException("The number of query instances should be greater than or equal to 10")
        if cf_examples_list is not None:
            if any([len(cf_examples.final_cfs_df) < 10 for cf_examples in cf_examples_list]):
                raise UserConfigValidationException("The number of counterfactuals generated per query instance should be greater than or equal to 10")
        elif total_CFs < 10:
            raise UserConfigValidationException("The number of counterfactuals generated per query instance should be greater than or equal to 10")
        importances = self.feature_importance(query_instances,
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
        :param other_parameters: These are the same as the
        generate_counterfactuals method.

        :returns: An object of class CounterfactualExplanations that includes
        the list of counterfactuals per input, local feature importances per
        input, and the global feature importance summarized over all inputs.
        """

        if cf_examples_list is None:
            cf_examples_list = self.generate_counterfactuals(query_instances, total_CFs,
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

        # Summarizing the found counterfactuals
        for i in range(len(cf_examples_list)):
            cf_examples = cf_examples_list[i]
            org_instance = cf_examples.test_instance_df

            if cf_examples.final_cfs_df_sparse is not None:
                df = cf_examples.final_cfs_df_sparse
            else:
                df = cf_examples.final_cfs_df
            for index, row in df.iterrows():
                for col in self.data_interface.continuous_feature_names:
                    if not np.isclose(org_instance[col].iloc[0], row[col]):
                        if summary_importance is not None:
                            summary_importance[col] += 1
                        if local_importances is not None:
                            local_importances[i][col] += 1
                for col in self.data_interface.categorical_feature_names:
                    if org_instance[col].iloc[0] != row[col]:
                        if summary_importance is not None:
                            summary_importance[col] += 1
                        if local_importances is not None:
                            local_importances[i][col] += 1

            if local_importances is not None:
                for col in allcols:
                    local_importances[i][col] /= (cf_examples_list[0].final_cfs_df.shape[0])
        if summary_importance is not None:
            for col in allcols:
                summary_importance[col] /= (cf_examples_list[0].final_cfs_df.shape[0]*len(cf_examples_list))
        return CounterfactualExplanations(cf_examples_list,
                local_importance=local_importances,
                summary_importance=summary_importance)


    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        return self.model.get_output(input_instance)

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: Final CFs in original user-fed format, in a pandas dataframe.
        :param query_instance: Query instance in original user-fed format, in a pandas dataframe.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.
        """
        # :param total_random_inits: total random initialization required for algorithm = RandomInitCF (where k CFs are generated by running a CF generation algorithm k times with random initializations.)

        # resetting index to make sure .loc works
        final_cfs_sparse = final_cfs_sparse.reset_index(drop=True)
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
        for cf_ix in range(len(final_cfs_sparse)):
            current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
            for feature in features_sorted:
                #current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
                #feat_ix = self.data_interface.continuous_feature_names.index(feature)
                diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]
                if(abs(diff) <= quantiles[feature]):
                    if posthoc_sparsity_algorithm == "linear":
                        final_cfs_sparse = self.do_linear_search(diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

                    elif posthoc_sparsity_algorithm == "binary":
                        final_cfs_sparse = self.do_binary_search(diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred)

            temp_preds = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
            #temp_preds = np.round(temp_preds.flatten().tolist(), 3)[0]
            cfs_preds_sparse.append(temp_preds)

        final_cfs_sparse[self.data_interface.outcome_name] = self.get_model_output_from_scores(cfs_preds_sparse)
        #final_cfs_sparse[self.data_interface.outcome_name] = np.round(final_cfs_sparse[self.data_interface.outcome_name], 3)
        return final_cfs_sparse

    def do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred_orig):
        """Performs a greedy linear search - moves the continuous features in CFs towards original values in query_instance greedily until the prediction class changes."""

        old_diff = diff
        change = (10**-decimal_prec[feature]) # the minimal possible change for a feature
        current_pred = current_pred_orig
        if self.model.model_type == 'classifier':
            while((abs(diff)>10e-4) and (np.sign(diff*old_diff) > 0) and self.is_cf_valid(current_pred)):
                old_val = final_cfs_sparse.iloc[cf_ix][feature]
                final_cfs_sparse.loc[cf_ix, feature] += np.sign(diff)*change
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])
                old_diff = diff

                if not self.is_cf_valid(current_pred):
                    final_cfs_sparse.loc[cf_ix, feature] = old_val
                    diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]
                    return final_cfs_sparse

                diff = query_instance[feature].iloc[0] - final_cfs_sparse.iloc[cf_ix][feature]

        return final_cfs_sparse

    def do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feature, final_cfs_sparse, current_pred):
        """Performs a binary search between continuous features of a CF and corresponding values in query_instance until the prediction class changes."""

        old_val = final_cfs_sparse.iloc[cf_ix][feature]
        final_cfs_sparse.loc[cf_ix, feature] = query_instance[feature].iloc[0]
        # Prediction of the query instance
        current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

        # first check if assigning query_instance values to a CF is required.
        if self.is_cf_valid(current_pred):
            return final_cfs_sparse
        else:
            final_cfs_sparse.loc[cf_ix, feature] = old_val

        # move the CF values towards the query_instance
        if diff > 0:
            left = final_cfs_sparse.iloc[cf_ix][feature]
            right = query_instance[feature].iloc[0]

            while left <= right:
                current_val = left + ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.loc[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.is_cf_valid(current_pred):
                    left = current_val + (10 ** -decimal_prec[feature])
                else:
                    right = current_val - (10 ** -decimal_prec[feature])

        else:
            left = query_instance[feature].iloc[0]
            right = final_cfs_sparse.iloc[cf_ix][feature]

            while right >= left:
                current_val = right - ((right - left)/2)
                current_val = round(current_val, decimal_prec[feature])

                final_cfs_sparse.loc[cf_ix, feature] = current_val
                current_pred = self.predict_fn_for_sparsity(final_cfs_sparse.iloc[[cf_ix]][self.data_interface.feature_names])

                if current_val == right or current_val == left:
                    break

                if self.is_cf_valid(current_pred):
                    right = current_val - (10**-decimal_prec[feature])
                else:
                    left = current_val + (10**-decimal_prec[feature])

        return final_cfs_sparse

    def infer_target_cfs_class(self, desired_class_input, original_pred,
            num_output_nodes):
        """ Infer the target class for generating CFs. Only called when
        model_type=="classifier".
        TODO: Add support for opposite desired class in multiclass. Downstream methods should decide
        whether it is allowed or not.
        """
        target_class = None
        if desired_class_input == "opposite":
            if num_output_nodes == 2:
                original_pred_1 = np.argmax(original_pred)
                target_class = int(1 - original_pred_1)

            elif num_output_nodes > 2:
                raise ValueError("Desired class cannot be opposite if the number of classes is more than 2.")

        if isinstance(desired_class_input, int):
            if desired_class_input >= 0 and desired_class_input < num_output_nodes:
                target_class = desired_class_input
            else:
                raise ValueError("Desired class should be within 0 and num_classes-1.")
        return target_class

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
            if self.model.model_type == "classifier":
                if self.num_output_nodes == 2: # binary
                    pred_1 = pred[self.num_output_nodes-1]
                    validity[i] = 1 if ((self.target_cf_class == 0 and pred_1<= self.stopping_threshold) or (self.target_cf_class == 1 and pred_1>= self.stopping_threshold)) else 0
                else: # multiclass
                    if np.argmax(pred) == self.target_cf_class:
                        validity[i] = 1
            elif self.model.model_type == "regressor":
                if self.target_cf_range[0] <= pred <= self.target_cf_range[1]:
                    validity[i] = 1
        return validity

    def is_cf_valid(self, model_score):
        """Check if a cf belongs to the target class or target range.
        """
        # Converting to single prediction if the prediction is provided as a
        # singleton array
        correct_dim = 1 if self.model.model_type == "classifier" else 0
        if hasattr(model_score, "shape") and len(model_score.shape) > correct_dim:
            model_score = model_score[0]
        # Converting target_cf_class to a scalar (tf/torch have it as (1,1) shape)
        target_cf_class = self.target_cf_class
        if hasattr(self.target_cf_class, "shape"):
            if len(self.target_cf_class.shape) == 1:
                target_cf_class = self.target_cf_class[0]
            elif len(self.target_cf_class.shape) == 2:
                target_cf_class = self.target_cf_class[0][0]
        target_cf_class = int(target_cf_class)

        if self.model.model_type == "classifier":
            if self.num_output_nodes == 1:  # for tensorflow/pytorch models
                pred_1 = model_score[0]
                validity = True if ((target_cf_class == 0 and pred_1<= self.stopping_threshold) or (target_cf_class == 1 and pred_1>= self.stopping_threshold)) else False
                return validity
            if self.num_output_nodes == 2:  # binary
                pred_1 = model_score[self.num_output_nodes-1]
                validity = True if ((target_cf_class == 0 and pred_1<= self.stopping_threshold) or (target_cf_class == 1 and pred_1>= self.stopping_threshold)) else False
                return validity
            else:  # multiclass
                if np.argmax(model_score) == target_cf_class:
                    return True
        elif self.model.model_type == "regressor":
            if self.target_cf_range[0] <= model_score <= self.target_cf_range[1]:
                return True
            else:
                return False

    def get_model_output_from_scores(self, model_scores):
        if self.model.model_type == "classifier":
            output_type = np.int32
        else:
            output_type = np.float32
        model_output = np.zeros(len(model_scores), dtype=output_type)
        for i in range(len(model_scores)):
            if self.model.model_type == "classifier":
                model_output[i] = np.argmax(model_scores[i])
            elif self.model.model_type == "regressor":
                model_output[i] = model_scores[i]
        return model_output

    def check_permitted_range(self, permitted_range): # TODO: add comments as to where this is used if this function is necessary, else remove.
        """checks permitted range for continuous features"""
        if permitted_range is not None:
        #     if not self.data_interface.check_features_range(permitted_range):
        #         raise ValueError(
        #             "permitted range of features should be within their original range")
        #     else:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

    def check_mad_validity(self, feature_weights): # TODO: add comments as to where this is used if this function is necessary, else remove.
        """checks feature MAD validity and throw warnings"""
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

    def sigmoid(self, z): # used in VAE-based CF explainers
            return 1 / (1 + np.exp(-z))

    def build_KD_tree(self, data_df_copy, desired_range, desired_class, predicted_outcome_name):
        # Stores the predictions on the training data
        dataset_instance = self.data_interface.prepare_query_instance(
            query_instance=data_df_copy[self.data_interface.feature_names])

        predictions = self.model.model.predict(dataset_instance)
        # TODO: Is it okay to insert a column in the original dataframe with the predicted outcome? This is memory-efficient
        data_df_copy[predicted_outcome_name] = predictions

        # segmenting the dataset according to outcome
        dataset_with_predictions = None
        if self.model.model_type == 'classifier':
            dataset_with_predictions = data_df_copy.loc[[i == desired_class for i in predictions]].copy()

        elif self.model.model_type == 'regressor':
            dataset_with_predictions = data_df_copy.loc[
                [desired_range[0] <= pred <= desired_range[1] for pred in predictions]].copy()

        KD_tree = None
        # Prepares the KD trees for DiCE
        if len(dataset_with_predictions) > 0:
            dummies = pd.get_dummies(dataset_with_predictions[self.data_interface.feature_names])
            KD_tree = KDTree(dummies)

        return dataset_with_predictions, KD_tree, predictions
