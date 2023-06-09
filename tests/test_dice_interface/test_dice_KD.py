import numpy as np
import pytest

import dice_ml
from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.utils import helpers


@pytest.fixture(scope='session')
def KD_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
def KD_binary_vars_classification_exp_object(load_custom_vars_testing_dataset):
    backend = 'sklearn'
    dataset = load_custom_vars_testing_dataset
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_vars_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
def KD_multi_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_multiclass()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
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

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize(("desired_range", "desired_class", "total_CFs", "features_to_vary", "permitted_range"),
                             [(None, 0, 4, ['Numerical'], {'Categorical': ['b', 'c']})])
    def test_invalid_query_instance(self, desired_range, desired_class, sample_custom_query_1, total_CFs,
                                    features_to_vary, permitted_range):
        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        with pytest.raises(
                ValueError, match="is outside the permitted range and isn't allowed to vary"):
            self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, total_CFs=total_CFs,
                                               features_to_vary=features_to_vary, permitted_range=permitted_range)

    # Verifying the output of the KD tree
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 1)])
    @pytest.mark.parametrize(("posthoc_sparsity_algorithm"), ['linear', 'binary', None])
    def test_KD_tree_output(self, desired_class, sample_custom_query_1, total_CFs, posthoc_sparsity_algorithm):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, desired_class=desired_class,
                                           total_CFs=total_CFs,
                                           posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[0])
        assert all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[0])

    # Verifying the output of the KD tree
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 1)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_class, sample_custom_query_1, total_CFs):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize(("desired_class", "total_CFs", "features_to_vary"), [(0, 1, ["Numerical"])])
    def test_features_to_vary(self, desired_class, sample_custom_query_2, total_CFs, features_to_vary):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, features_to_vary=features_to_vary)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[1])
        assert all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[1])

    # Testing that the permitted_range argument actually varies the features only within the permitted_range
    @pytest.mark.parametrize(("desired_class", "total_CFs", "permitted_range"), [(0, 1, {'Numerical': [1000, 10000]})])
    def test_permitted_range(self, desired_class, sample_custom_query_2, total_CFs, permitted_range):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, permitted_range=permitted_range)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df
        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[1])
        assert all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[1])

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize(("desired_class", "total_CFs", "permitted_range"), [(0, 4, {'Categorical': ['b', 'c']})])
    def test_permitted_range_categorical(self, desired_class, sample_custom_query_2, total_CFs, permitted_range):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, permitted_range=permitted_range)
        assert all(i in permitted_range["Categorical"] for i in self.exp.final_cfs_df.Categorical.values)

    # Ensuring that there are no duplicates in the resulting counterfactuals even if the dataset has duplicates
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 2)])
    def test_duplicates(self, desired_class, sample_custom_query_4, total_CFs):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                           desired_class=desired_class)

        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        self.exp.final_cfs_df = self.exp.final_cfs_df.reset_index(drop=True)

        expected_output = self.exp.data_interface.data_df.iloc[np.r_[2, 0]][self.exp.data_interface.feature_names]
        expected_output = expected_output.reset_index(drop=True)

        assert all(self.exp.final_cfs_df == expected_output)

    # Testing for index returned
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 1)])
    @pytest.mark.parametrize(("posthoc_sparsity_algorithm"), ['linear', 'binary', None])
    def test_index(self, desired_class, sample_custom_query_index, total_CFs, posthoc_sparsity_algorithm):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_index, total_CFs=total_CFs,
                                           desired_class=desired_class,
                                           posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        assert self.exp.final_cfs_df.index[0] == 3


class TestDiceKDMultiClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_multi_classification_exp_object):
        self.exp_multi = KD_multi_classification_exp_object  # explainer object
        self.data_df_copy = self.exp_multi.data_interface.data_df.copy()

    # Testing that the output of multiclass classification lies in the desired_class
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(2, 3)])
    @pytest.mark.parametrize(("posthoc_sparsity_algorithm"), ['linear', 'binary', None])
    def test_KD_tree_output(self, desired_class, sample_custom_query_2, total_CFs,
                            posthoc_sparsity_algorithm):
        self.exp_multi._generate_counterfactuals(query_instance=sample_custom_query_2, total_CFs=total_CFs,
                                                 desired_class=desired_class,
                                                 posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        assert all(i == desired_class for i in self.exp_multi.cfs_preds)


class TestDiceKDRegressionMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_regression_exp_object):
        self.exp_regr = KD_regression_exp_object  # explainer object
        self.data_df_copy = self.exp_regr.data_interface.data_df.copy()

    # Testing that the output of regression lies in the desired_range
    @pytest.mark.parametrize(("desired_range", "total_CFs"), [([1, 2.8], 6)])
    @pytest.mark.parametrize(("version"), ['2.0', '1.0'])
    @pytest.mark.parametrize(("posthoc_sparsity_algorithm"), ['linear', 'binary', None])
    def test_KD_tree_output(self, desired_range, sample_custom_query_2, total_CFs, version, posthoc_sparsity_algorithm):
        cf_examples = self.exp_regr._generate_counterfactuals(query_instance=sample_custom_query_2, total_CFs=total_CFs,
                                                              desired_range=desired_range,
                                                              posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        assert all(desired_range[0] <= i <= desired_range[1] for i in self.exp_regr.cfs_preds)

        assert cf_examples is not None
        json_str = cf_examples.to_json(version)
        assert json_str is not None

        recovered_cf_examples = CounterfactualExamples.from_json(json_str)
        assert recovered_cf_examples is not None
        assert cf_examples == recovered_cf_examples

    @pytest.mark.parametrize(("desired_range", "total_CFs"), [([1, 2.8], 6)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_range, sample_custom_query_2,
                                                        total_CFs):
        counterfactual_explanations = self.exp_regr.generate_counterfactuals(
                                            query_instances=sample_custom_query_2, total_CFs=total_CFs,
                                            desired_range=desired_range)

        assert counterfactual_explanations is not None
        json_str = counterfactual_explanations.to_json()
        assert json_str is not None

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_str)
        assert recovered_counterfactual_explanations is not None
        assert counterfactual_explanations == recovered_counterfactual_explanations

    # Testing for 0 CFs needed
    @pytest.mark.parametrize(("desired_class", "desired_range", "total_CFs"), [(0, [1, 2.8], 0)])
    def test_zero_cfs(self, desired_class, desired_range, sample_custom_query_4, total_CFs):
        self.exp_regr._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                                desired_range=desired_range)


class TestDiceKDBinaryVarsClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_binary_vars_classification_exp_object):
        self.exp = KD_binary_vars_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize(("desired_range", "desired_class", "total_CFs", "features_to_vary", "permitted_range"),
                             [(None, 0, 4, ['Numerical'], {'CategoricalNum': ['1', '2']})])
    def test_invalid_query_instance(self, desired_range, desired_class, sample_custom_vars_query_1, total_CFs,
                                    features_to_vary, permitted_range):
        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        with pytest.raises(ValueError, match="is outside the permitted range and isn't allowed to vary"):
            self.exp._generate_counterfactuals(query_instance=sample_custom_vars_query_1, total_CFs=total_CFs,
                                               features_to_vary=features_to_vary, permitted_range=permitted_range)

    # Verifying the output of the KD tree
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 1)])
    @pytest.mark.parametrize('posthoc_sparsity_algorithm', ['linear', 'binary', None])
    def test_KD_tree_output(self, desired_class, sample_custom_vars_query_1, total_CFs, posthoc_sparsity_algorithm):
        self.exp._generate_counterfactuals(query_instance=sample_custom_vars_query_1, desired_class=desired_class,
                                           total_CFs=total_CFs,
                                           posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[0])
        assert all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[0])

    # Verifying the output of the KD tree
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 1)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_class, sample_custom_vars_query_1, total_CFs):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_vars_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
