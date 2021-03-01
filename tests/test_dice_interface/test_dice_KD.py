import pytest
import numpy as np
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from sklearn.neighbors import KDTree


@pytest.fixture
def KD_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture
def KD_multi_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_multiclass()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp

@pytest.fixture
def KD_regression_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_regression()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


class TestDiceKDBinaryClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_binary_classification_exp_object):
        self.exp = KD_binary_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    # When no elements in the desired_class are present
    @pytest.mark.parametrize("desired_range, desired_class, features_to_vary, total_CFs", [(None, 7, "all", 3)])
    def test_empty_KD(self, desired_range, desired_class, features_to_vary, sample_custom_query_1, total_CFs):
        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_1.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_1)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_1,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range=None,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 4, ['Numerical'], {'Categorical': ['b', 'c']})])
    def test_invalid_query_instance(self, desired_range, desired_class, sample_custom_query_1, total_CFs,
                                    features_to_vary, permitted_range):
        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        try:
            self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, total_CFs=total_CFs,
                                               features_to_vary=features_to_vary, permitted_range=permitted_range)
            assert False
        except ValueError:
            assert True

    # Verifying the output of the KD tree
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 1, "all", None)])
    def test_KD_tree_output(self, desired_range, desired_class, sample_custom_query_1, total_CFs, features_to_vary,
                            permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range
        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_1, feature_ranges_orig)

        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_1.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_1)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_1,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)
        final_cfs.Numerical = final_cfs.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(final_cfs.Numerical == expected_output.Numerical[0]) and \
               all(final_cfs.Categorical == expected_output.Categorical[0])

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 1, ["Numerical"], None)])
    def test_features_to_vary(self, desired_range, desired_class, sample_custom_query_2, total_CFs, features_to_vary,
                              permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range
        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_2, feature_ranges_orig)

        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_2.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_2)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_2,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)
        final_cfs.Numerical = final_cfs.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(final_cfs.Numerical == expected_output.Numerical[1]) and \
               all(final_cfs.Categorical == expected_output.Categorical[1])

    # Testing that the permitted_range argument actually varies the features only within the permitted_range
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 1, "all", {'Numerical': [1000, 10000]})])
    def test_permitted_range(self, desired_range, desired_class, sample_custom_query_2, total_CFs, features_to_vary,
                             permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range
        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_2, feature_ranges_orig)

        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_2.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_2)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_2,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)
        final_cfs.Numerical = final_cfs.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(final_cfs.Numerical == expected_output.Numerical[1]) and \
               all(final_cfs.Categorical == expected_output.Categorical[1])

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 4, "all", {'Categorical': ['b', 'c']})])
    def test_permitted_range_categorical(self, desired_range, desired_class, sample_custom_query_2, total_CFs,
                                         features_to_vary,
                                         permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range

        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_2, feature_ranges_orig)

        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_2.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_2)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_2,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)
        assert all(i in permitted_range["Categorical"] for i in final_cfs.Categorical.values)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 1, "all", None)])
    def test_query_instance_outside_bounds(self, desired_range, desired_class, sample_custom_query_3, total_CFs,
                                           features_to_vary,
                                           permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range

        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        try:
            self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_3, feature_ranges_orig)
            assert False
        except ValueError:
            assert True

    # Ensuring that there are no duplicates in the resulting counterfactuals even if the dataset has duplicates
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 2, "all", None)])
    def test_duplicates(self, desired_range, desired_class, sample_custom_query_4, total_CFs, features_to_vary,
                        permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp.feature_range = self.exp.data_interface.permitted_range
            feature_ranges_orig = self.exp.feature_range
        else:  # compute the new ranges based on user input
            self.exp.feature_range, feature_ranges_orig = self.exp.data_interface.get_features_range(permitted_range)

        self.exp.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_4, feature_ranges_orig)

        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_4.copy()
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_custom_query_4)

        # find the predicted value of query_instance
        test_pred = self.exp.predict_fn(query_instance)[0]

        query_instance[self.exp.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp.find_counterfactuals(self.data_df_copy,
                                                                             sample_custom_query_4,
                                                                             sample_custom_query_orig,
                                                                             desired_range,
                                                                             desired_class,
                                                                             total_CFs, features_to_vary,
                                                                             permitted_range,
                                                                             sparsity_weight=1,
                                                                             stopping_threshold=0.5,
                                                                             posthoc_sparsity_param=0.1,
                                                                             posthoc_sparsity_algorithm='binary',
                                                                             verbose=False)

        final_cfs.Numerical = final_cfs.Numerical.astype(int)
        final_cfs = final_cfs.reset_index(drop=True)

        expected_output = self.exp.data_interface.data_df.iloc[np.r_[2, 0]][self.exp.data_interface.feature_names]
        expected_output = expected_output.reset_index(drop=True)

        assert all(final_cfs == expected_output)


class TestDiceKDMultiClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_multi_classification_exp_object):
        self.exp_multi = KD_multi_classification_exp_object  # explainer object
        self.data_df_copy = self.exp_multi.data_interface.data_df.copy()

    # Testing that the output of multiclass classification lies in the desired_class
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 2, 3, "all", None)])
    def test_KD_tree_output(self, desired_range, desired_class, sample_custom_query_2, total_CFs, features_to_vary,
                            permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp_multi.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp_multi.feature_range = self.exp_multi.data_interface.permitted_range
            feature_ranges_orig = self.exp_multi.feature_range
        else:  # compute the new ranges based on user input
            self.exp_multi.feature_range, feature_ranges_orig = self.exp_multi.data_interface.get_features_range(permitted_range)

        self.exp_multi.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_2, feature_ranges_orig)
        predictions = [0, 2, 1, 0, 2]
        predicted_outcome_name = self.exp_multi.data_interface.outcome_name + '_pred'
        self.data_df_copy[predicted_outcome_name] = predictions

        # segmenting the dataset according to outcome
        dataset_with_predictions = None
        if self.exp_multi.model.model_type == 'classifier':
            dataset_with_predictions = self.data_df_copy.loc[[i == desired_class for i in predictions]].copy()

        elif self.exp_multi.model.model_type == 'regressor':
            dataset_with_predictions = self.data_df_copy.loc[
                [desired_range[0] <= pred <= desired_range[1] for pred in predictions]].copy()

        KD_tree = None
        # Prepares the KD trees for DiCE
        if len(dataset_with_predictions) > 0:
            dummies = pd.get_dummies(dataset_with_predictions[self.exp_multi.data_interface.feature_names])
            KD_tree = KDTree(dummies)

        self.exp_multi.dataset_with_predictions = dataset_with_predictions
        self.exp_multi.KD_tree = KD_tree
        self.exp_multi.predictions = predictions

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_2.copy()
        query_instance = self.exp_multi.data_interface.prepare_query_instance(query_instance=sample_custom_query_2)

        # find the predicted value of query_instance
        test_pred = self.exp_multi.predict_fn(query_instance)[0]

        query_instance[self.exp_multi.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp_multi.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp_multi.find_counterfactuals(self.data_df_copy,
                                                                                   sample_custom_query_2,
                                                                                   sample_custom_query_orig,
                                                                                   desired_range,
                                                                                   desired_class,
                                                                                   total_CFs, features_to_vary,
                                                                                   permitted_range,
                                                                                   sparsity_weight=1,
                                                                                   stopping_threshold=0.5,
                                                                                   posthoc_sparsity_param=0.1,
                                                                                   posthoc_sparsity_algorithm='binary',
                                                                                   verbose=False)
        assert all(i == desired_class for i in cfs_preds)


class TestDiceKDRegressionMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_regression_exp_object):
        self.exp_regr = KD_regression_exp_object  # explainer object
        self.data_df_copy = self.exp_regr.data_interface.data_df.copy()

    # Testing that the output of regression lies in the desired_range
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [([1, 2.8], "opposite", 6, "all", None)])
    def test_KD_tree_output(self, desired_range, desired_class, sample_custom_query_2, total_CFs, features_to_vary,
                            permitted_range):
        if features_to_vary == 'all':
            features_to_vary = self.exp_regr.data_interface.feature_names

        if permitted_range is None:  # use the precomputed default
            self.exp_regr.feature_range = self.exp_regr.data_interface.permitted_range
            feature_ranges_orig = self.exp_regr.feature_range
        else:  # compute the new ranges based on user input
            self.exp_regr.feature_range, feature_ranges_orig = self.exp_regr.data_interface.get_features_range(permitted_range)

        self.exp_regr.check_query_instance_validity(features_to_vary, permitted_range, sample_custom_query_2, feature_ranges_orig)

        # Stores the predictions on the training data
        dataset_instance = self.exp_regr.data_interface.prepare_query_instance(
            query_instance=self.data_df_copy[self.exp_regr.data_interface.feature_names])

        predictions = [1, 2.8, 0.8, 22, 1.7]
        predicted_outcome_name = self.exp_regr.data_interface.outcome_name + '_pred'
        self.data_df_copy[predicted_outcome_name] = predictions

        # segmenting the dataset according to outcome
        dataset_with_predictions = None
        if self.exp_regr.model.model_type == 'classifier':
            dataset_with_predictions = self.data_df_copy.loc[[i == desired_class for i in predictions]].copy()

        elif self.exp_regr.model.model_type == 'regressor':
            dataset_with_predictions = self.data_df_copy.loc[
                [desired_range[0] <= pred <= desired_range[1] for pred in predictions]].copy()

        KD_tree = None
        # Prepares the KD trees for DiCE
        if len(dataset_with_predictions) > 0:
            dummies = pd.get_dummies(dataset_with_predictions[self.exp_regr.data_interface.feature_names])
            KD_tree = KDTree(dummies)

        self.exp_regr.dataset_with_predictions = dataset_with_predictions
        self.exp_regr.KD_tree = KD_tree
        self.exp_regr.predictions = predictions

        # Prepares user defined query_instance for DiCE.
        sample_custom_query_orig = sample_custom_query_2.copy()
        query_instance = self.exp_regr.data_interface.prepare_query_instance(query_instance=sample_custom_query_2)

        # find the predicted value of query_instance
        test_pred = self.exp_regr.predict_fn(query_instance)[0]

        query_instance[self.exp_regr.data_interface.outcome_name] = test_pred

        if features_to_vary == 'all':
            features_to_vary = self.exp_regr.data_interface.feature_names

        query_instance, final_cfs, cfs_preds = self.exp_regr.find_counterfactuals(self.data_df_copy,
                                                                                   sample_custom_query_2,
                                                                                   sample_custom_query_orig,
                                                                                   desired_range,
                                                                                   desired_class,
                                                                                   total_CFs, features_to_vary,
                                                                                   permitted_range,
                                                                                   sparsity_weight=1,
                                                                                   stopping_threshold=0.5,
                                                                                   posthoc_sparsity_param=0.1,
                                                                                   posthoc_sparsity_algorithm='binary',
                                                                                   verbose=False)
        assert all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds)