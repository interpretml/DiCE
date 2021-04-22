import json
import pytest

import dice_ml
from dice_ml.utils import helpers

from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.utils.exception import UserConfigValidationException


class TestCounterfactualExplanations:

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def test_serialization_deserialization_counterfactual_explanations_class(self, version):

        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance=None,
            version=version)
        assert counterfactual_explanations.cf_examples_list is not None
        assert len(counterfactual_explanations.cf_examples_list) == 0
        assert counterfactual_explanations.summary_importance is None
        assert counterfactual_explanations.local_importance is None
        assert counterfactual_explanations.metadata is not None
        assert counterfactual_explanations.metadata['version'] is not None
        assert counterfactual_explanations.metadata['version'] == version

        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        assert counterfactual_explanations_as_json is not None

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)

        assert recovered_counterfactual_explanations is not None
        assert recovered_counterfactual_explanations.metadata['version'] == version
        assert counterfactual_explanations == recovered_counterfactual_explanations

    def test_sorted_summary_importance_counterfactual_explanations(self):

        unsorted_summary_importance = {
            "age": 0.985,
            "workclass": 0.65,
            "education": 0.915,
            "occupation": 0.95,
            "hours_per_week": 0.985,
            "gender": 0.67,
            "marital_status": 0.655,
            "race": 0.41
        }

        sorted_summary_importance = {
            "age": 0.985,
            "hours_per_week": 0.985,
            "occupation": 0.95,
            "education": 0.915,
            "gender": 0.67,
            "marital_status": 0.655,
            "workclass": 0.65,
            "race": 0.41
        }

        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance=unsorted_summary_importance)

        assert unsorted_summary_importance == counterfactual_explanations.summary_importance
        assert sorted_summary_importance == counterfactual_explanations.summary_importance

        assert list(unsorted_summary_importance.keys()) != list(counterfactual_explanations.summary_importance.keys())
        assert list(sorted_summary_importance.keys()) == list(counterfactual_explanations.summary_importance.keys())

    def test_sorted_local_importance_counterfactual_explanations(self):

        unsorted_local_importance = [
            {
                "age": 0.985,
                "workclass": 0.65,
                "education": 0.915,
                "occupation": 0.95,
                "hours_per_week": 0.985,
                "gender": 0.67,
                "marital_status": 0.655,
                "race": 0.41
            },
            {
                "age": 0.985,
                "workclass": 0.65,
                "education": 0.915,
                "occupation": 0.95,
                "hours_per_week": 0.985,
                "gender": 0.67,
                "marital_status": 0.655,
                "race": 0.41
            }
        ]

        sorted_local_importance = [
            {
                "age": 0.985,
                "hours_per_week": 0.985,
                "occupation": 0.95,
                "education": 0.915,
                "gender": 0.67,
                "marital_status": 0.655,
                "workclass": 0.65,
                "race": 0.41
            },
            {
                "age": 0.985,
                "hours_per_week": 0.985,
                "occupation": 0.95,
                "education": 0.915,
                "gender": 0.67,
                "marital_status": 0.655,
                "workclass": 0.65,
                "race": 0.41
            }
        ]         

        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=unsorted_local_importance,
            summary_importance=None)

        for index in range(0, len(unsorted_local_importance)):
            assert unsorted_local_importance[index] == counterfactual_explanations.local_importance[index]
            assert sorted_local_importance[index] == counterfactual_explanations.local_importance[index]

        for index in range(0, len(unsorted_local_importance)):
            assert list(unsorted_local_importance[index].keys()) != list(counterfactual_explanations.local_importance[index].keys())
            assert list(sorted_local_importance[index].keys()) == list(counterfactual_explanations.local_importance[index].keys())

    @pytest.mark.parametrize('version', ['3.0', ''])
    def test_unsupported_versions_json_input(self, version):
        json_str = json.dumps({'metadata': {'version': version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "Incompatible version {} found in json input".format(version) in str(ucve)

        json_str = json.dumps({'metadata': {'versio': version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "No version field in the json input" in str(ucve)



@pytest.fixture
def random_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


class TestSerializationCounterfactualExplanations:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, random_binary_classification_exp_object):
        self.exp = random_binary_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 2)])
    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def test_random_counterfactual_explanations_output(self, desired_class,
                                                       sample_custom_query_1, total_CFs,
                                                       version):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
        assert len(counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert counterfactual_explanations.local_importance is None
        assert counterfactual_explanations.summary_importance is None

        counterfactual_explanations.metadata['version'] = version
        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        assert recovered_counterfactual_explanations is not None
        assert recovered_counterfactual_explanations == counterfactual_explanations
        assert recovered_counterfactual_explanations.metadata['version'] == version

        assert len(recovered_counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert recovered_counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert recovered_counterfactual_explanations.local_importance is None
        assert recovered_counterfactual_explanations.summary_importance is None

    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 10)])
    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def test_random_local_importance_output(self, desired_class, sample_custom_query_1,
                                            total_CFs, version):
        counterfactual_explanations = self.exp.local_feature_importance(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
        assert len(counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert counterfactual_explanations.local_importance is not None
        assert counterfactual_explanations.summary_importance is None

        counterfactual_explanations.metadata['version'] = version
        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        assert recovered_counterfactual_explanations is not None
        assert recovered_counterfactual_explanations == counterfactual_explanations
        assert recovered_counterfactual_explanations.metadata['version'] == version

        assert len(recovered_counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert recovered_counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert recovered_counterfactual_explanations.local_importance is not None
        assert counterfactual_explanations.summary_importance is None

    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 10)])
    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def test_random_summary_importance_output(self, desired_class, sample_custom_query_10,
                                              total_CFs, version):
        counterfactual_explanations = self.exp.global_feature_importance(
            query_instances=sample_custom_query_10, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
        assert len(counterfactual_explanations.cf_examples_list) == sample_custom_query_10.shape[0]
        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert counterfactual_explanations.local_importance is not None
        assert counterfactual_explanations.summary_importance is not None

        counterfactual_explanations.metadata['version'] = version
        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        assert recovered_counterfactual_explanations is not None
        assert recovered_counterfactual_explanations == counterfactual_explanations
        assert recovered_counterfactual_explanations.metadata['version'] == version

        assert len(recovered_counterfactual_explanations.cf_examples_list) == sample_custom_query_10.shape[0]
        assert recovered_counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert recovered_counterfactual_explanations.local_importance is not None
        assert counterfactual_explanations.summary_importance is not None
