import pytest
import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException


@pytest.fixture
def random_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_binary()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


class TestDiceRandomBinaryClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, random_binary_classification_exp_object):
        self.exp = random_binary_classification_exp_object  # explainer object

    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    def test_random_counterfactual_explanations_output(self, desired_class, sample_custom_query_1, total_CFs):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
        assert len(counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs

    # When invalid desired_class is given
    @pytest.mark.parametrize("desired_class, desired_range, total_CFs, features_to_vary, permitted_range",
                             [(7, None, 3, "all", None)])
    def test_no_cfs(self, desired_class, desired_range, sample_custom_query_1, total_CFs, features_to_vary,
                    permitted_range):
        with pytest.raises(UserConfigValidationException):
            self.exp._generate_counterfactuals(features_to_vary=features_to_vary, query_instance=sample_custom_query_1,
                                               total_CFs=total_CFs,
                                               desired_class=desired_class, desired_range=desired_range,
                                               permitted_range=permitted_range)

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize("features_to_vary, permitted_range, feature_weights",
                             [(['Numerical'], {'Categorical': ['b', 'c']}, "inverse_mad")])
    def test_invalid_query_instance(self, sample_custom_query_1, features_to_vary, permitted_range, feature_weights):
        with pytest.raises(ValueError):
            self.exp.setup(features_to_vary, permitted_range, sample_custom_query_1, feature_weights)

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize("desired_class, desired_range, total_CFs, features_to_vary, permitted_range",
                             [(1, None, 2, ["Numerical"], None)])
    def test_features_to_vary(self, desired_class, desired_range, sample_custom_query_2, total_CFs, features_to_vary,
                              permitted_range):
        features_to_vary = self.exp.setup(features_to_vary, None, sample_custom_query_2, "inverse_mad")
        ans = self.exp._generate_counterfactuals(query_instance=sample_custom_query_2,
                                                 features_to_vary=features_to_vary,
                                                 total_CFs=total_CFs, desired_class=desired_class,
                                                 desired_range=desired_range, permitted_range=permitted_range)

        for feature in self.exp.data_interface.feature_names:
            if feature not in features_to_vary:
                assert all(ans.final_cfs_df[feature].values[i] == sample_custom_query_2[feature].values[0] for i in
                           range(total_CFs))

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize("desired_class, desired_range, total_CFs, permitted_range",
                             [(1, None, 2, {'Categorical': ['a', 'c']})])
    def test_permitted_range_categorical(self, desired_class, desired_range, total_CFs, sample_custom_query_2,
                                         permitted_range):
        features_to_vary = self.exp.setup("all", permitted_range, sample_custom_query_2, "inverse_mad")
        ans = self.exp._generate_counterfactuals(query_instance=sample_custom_query_2,
                                                 features_to_vary=features_to_vary, permitted_range=permitted_range,
                                                 total_CFs=total_CFs, desired_class=desired_class,
                                                 desired_range=desired_range)

        for feature in permitted_range:
            assert all(
                permitted_range[feature][0] <= ans.final_cfs_df[feature].values[i] <= permitted_range[feature][1] for i
                in range(total_CFs))
