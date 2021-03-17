import pytest
import numpy as np
import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.counterfactual_explanations import CounterfactualExplanations


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

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize("desired_class, total_CFs", [(1, 3), ('a', 3)])
    def test_unsupported_binary_class(self, desired_class, sample_custom_query_1, total_CFs):
        with pytest.raises(UserConfigValidationException) as ucve:
            self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, total_CFs=total_CFs,
                                               desired_class=desired_class)
        if desired_class == 1:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "The target class for {0} could not be identified".format(desired_class) in str(ucve)

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize("desired_range, desired_class, total_CFs, features_to_vary, permitted_range",
                             [(None, 0, 4, ['Numerical'], {'Categorical': ['b', 'c']})])
    def test_invalid_query_instance(self, desired_range, desired_class, sample_custom_query_1, total_CFs,
                                    features_to_vary, permitted_range):
        self.exp.dataset_with_predictions, self.exp.KD_tree, self.exp.predictions = \
            self.exp.build_KD_tree(self.data_df_copy, desired_range, desired_class, self.exp.predicted_outcome_name)

        with pytest.raises(ValueError):
            self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, total_CFs=total_CFs,
                                               features_to_vary=features_to_vary, permitted_range=permitted_range)

    # Verifying the output of the KD tree
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    def test_KD_tree_output(self, desired_class, sample_custom_query_1, total_CFs):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, desired_class=desired_class,
                                           total_CFs=total_CFs)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[0]) and \
               all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[0])

    # Verifying the output of the KD tree
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_class, sample_custom_query_1, total_CFs):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize("desired_class, total_CFs, features_to_vary", [(0, 1, ["Numerical"])])
    def test_features_to_vary(self, desired_class, sample_custom_query_2, total_CFs, features_to_vary):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, features_to_vary=features_to_vary)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[1]) and \
               all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[1])

    # Testing that the permitted_range argument actually varies the features only within the permitted_range
    @pytest.mark.parametrize("desired_class, total_CFs, permitted_range", [(0, 1, {'Numerical': [1000, 10000]})])
    def test_permitted_range(self, desired_class, sample_custom_query_2, total_CFs, permitted_range):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, permitted_range=permitted_range)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df
        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[1]) and \
               all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[1])

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize("desired_class, total_CFs, permitted_range", [(0, 4, {'Categorical': ['b', 'c']})])
    def test_permitted_range_categorical(self, desired_class, sample_custom_query_2, total_CFs, permitted_range):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, permitted_range=permitted_range)
        assert all(i in permitted_range["Categorical"] for i in self.exp.final_cfs_df.Categorical.values)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    def test_query_instance_outside_bounds(self, desired_class, sample_custom_query_3, total_CFs):
        with pytest.raises(ValueError):
            self.exp._generate_counterfactuals(query_instance=sample_custom_query_3, total_CFs=total_CFs,
                                               desired_class=desired_class)

    # Ensuring that there are no duplicates in the resulting counterfactuals even if the dataset has duplicates
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 2)])
    def test_duplicates(self, desired_class, sample_custom_query_4, total_CFs):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                           desired_class=desired_class)

        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        self.exp.final_cfs_df = self.exp.final_cfs_df.reset_index(drop=True)

        expected_output = self.exp.data_interface.data_df.iloc[np.r_[2, 0]][self.exp.data_interface.feature_names]
        expected_output = expected_output.reset_index(drop=True)

        assert all(self.exp.final_cfs_df == expected_output)

    # Testing for 0 CFs needed
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 0)])
    def test_zero_cfs(self, desired_class, sample_custom_query_4, total_CFs):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                           desired_class=desired_class)


class TestDiceKDMultiClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_multi_classification_exp_object):
        self.exp_multi = KD_multi_classification_exp_object  # explainer object
        self.data_df_copy = self.exp_multi.data_interface.data_df.copy()

    # Testing that the output of multiclass classification lies in the desired_class
    @pytest.mark.parametrize("desired_class, total_CFs", [(2, 3)])
    def test_KD_tree_output(self, desired_class, sample_custom_query_2, total_CFs):
        self.exp_multi._generate_counterfactuals(query_instance=sample_custom_query_2, total_CFs=total_CFs,
                                                 desired_class=desired_class)
        assert all(i == desired_class for i in self.exp_multi.cfs_preds)

    # Testing that the output of multiclass classification lies in the desired_class
    @pytest.mark.parametrize("desired_class, total_CFs", [(2, 3)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_class, sample_custom_query_2, total_CFs):
        counterfactual_explanations = self.exp_multi.generate_counterfactuals(
                                        query_instances=sample_custom_query_2, total_CFs=total_CFs,
                                        desired_class=desired_class)
        assert all(i == desired_class for i in self.exp_multi.cfs_preds)

        assert counterfactual_explanations is not None

    # Testing for 0 CFs needed
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 0)])
    def test_zero_cfs(self, desired_class, sample_custom_query_4, total_CFs):
        self.exp_multi._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                                 desired_class=desired_class)

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize("desired_class, total_CFs", [(100, 3), ('opposite', 3)])
    def test_unsupported_multiclass(self, desired_class, sample_custom_query_4, total_CFs):
        with pytest.raises(UserConfigValidationException) as ucve:
            self.exp_multi._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                                     desired_class=desired_class)
        if desired_class == 100:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "Desired class cannot be opposite if the number of classes is more than 2." in str(ucve)



class TestDiceKDRegressionMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_regression_exp_object):
        self.exp_regr = KD_regression_exp_object  # explainer object
        self.data_df_copy = self.exp_regr.data_interface.data_df.copy()

    # Testing that the output of regression lies in the desired_range
    @pytest.mark.parametrize("desired_range, total_CFs", [([1, 2.8], 6)])
    def test_KD_tree_output(self, desired_range, sample_custom_query_2, total_CFs):
        cf_examples = self.exp_regr._generate_counterfactuals(query_instance=sample_custom_query_2, total_CFs=total_CFs,
                                                              desired_range=desired_range)
        assert all(desired_range[0] <= i <= desired_range[1] for i in self.exp_regr.cfs_preds)

        assert cf_examples is not None
        json_str = cf_examples.to_json()
        assert json_str is not None

        recovered_cf_examples = CounterfactualExamples.from_json(json_str)
        assert recovered_cf_examples is not None
        assert cf_examples == recovered_cf_examples

    @pytest.mark.parametrize("desired_range, total_CFs", [([1, 2.8], 6)])
    def test_KD_tree_counterfactual_explanations_output(self, desired_range, sample_custom_query_2, total_CFs):
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
    @pytest.mark.parametrize("desired_class, desired_range, total_CFs", [(0, [1, 2.8], 0)])
    def test_zero_cfs(self, desired_class, desired_range, sample_custom_query_4, total_CFs):
        self.exp_regr._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                                desired_range=desired_range)
