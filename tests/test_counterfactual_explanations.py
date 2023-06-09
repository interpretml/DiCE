import json

import pytest
from raiutils.exceptions import UserConfigValidationException

import dice_ml
from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.utils import helpers


class TestCounterfactualExplanations:

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
            assert list(unsorted_local_importance[index].keys()) != \
                list(counterfactual_explanations.local_importance[index].keys())
            assert list(sorted_local_importance[index].keys()) == \
                list(counterfactual_explanations.local_importance[index].keys())


@pytest.fixture(scope='session')
def random_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_binary()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


class TestSerializationCounterfactualExplanations:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, random_binary_classification_exp_object):
        self.exp = random_binary_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def verify_counterfactual_explanations(self, counterfactual_explanations,
                                           total_CFs, num_query_points, version,
                                           local_importance_available=False,
                                           summary_importance_available=False):
        assert counterfactual_explanations is not None
        assert counterfactual_explanations.cf_examples_list is not None
        assert len(counterfactual_explanations.cf_examples_list) == num_query_points
        if total_CFs is not None:
            assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        assert counterfactual_explanations.metadata is not None
        assert counterfactual_explanations.metadata['version'] is not None
        counterfactual_explanations.metadata['version'] = version
        if local_importance_available:
            assert counterfactual_explanations.local_importance is not None
            assert len(counterfactual_explanations.local_importance) == num_query_points
        else:
            assert counterfactual_explanations.local_importance is None
        if summary_importance_available:
            assert counterfactual_explanations.summary_importance is not None
        else:
            assert counterfactual_explanations.summary_importance is None

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 2)])
    def test_counterfactual_explanations_output(self, desired_class,
                                                sample_custom_query_1, total_CFs,
                                                version):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_1.shape[0], version)

        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_1.shape[0], version)

        assert recovered_counterfactual_explanations == counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 10)])
    def test_local_importance_output(self, desired_class, sample_custom_query_1,
                                     total_CFs, version):
        counterfactual_explanations = self.exp.local_feature_importance(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_1.shape[0], version,
                                                local_importance_available=True)

        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_1.shape[0], version,
                                                local_importance_available=True)

        assert recovered_counterfactual_explanations == counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 10)])
    def test_summary_importance_output(self, desired_class, sample_custom_query_10,
                                       total_CFs, version):
        counterfactual_explanations = self.exp.global_feature_importance(
            query_instances=sample_custom_query_10, desired_class=desired_class,
            total_CFs=total_CFs)

        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_10.shape[0], version,
                                                local_importance_available=True,
                                                summary_importance_available=True)

        json_output = counterfactual_explanations.to_json()
        assert json_output is not None
        assert json.loads(json_output).get('metadata').get('version') == version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        self.verify_counterfactual_explanations(counterfactual_explanations, total_CFs,
                                                sample_custom_query_10.shape[0], version,
                                                local_importance_available=True,
                                                summary_importance_available=True)

        assert recovered_counterfactual_explanations == counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    def test_empty_counterfactual_explanations_object(self, version):

        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance=None,
            version=version)
        self.verify_counterfactual_explanations(counterfactual_explanations, None,
                                                0, version)

        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        assert counterfactual_explanations_as_json is not None

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)

        self.verify_counterfactual_explanations(recovered_counterfactual_explanations, None,
                                                0, version)

        assert counterfactual_explanations == recovered_counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 2)])
    def test_no_counterfactuals_found(self, desired_class,
                                      sample_custom_query_1, total_CFs,
                                      version):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)
        counterfactual_explanations.cf_examples_list[0].final_cfs_df = None
        counterfactual_explanations.cf_examples_list[0].final_cfs_df_sparse = None
        self.verify_counterfactual_explanations(counterfactual_explanations, None,
                                                sample_custom_query_1.shape[0], version)
        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)
        self.verify_counterfactual_explanations(recovered_counterfactual_explanations, None,
                                                sample_custom_query_1.shape[0], version)
        assert counterfactual_explanations == recovered_counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 10)])
    def test_no_counterfactuals_found_local_importance(self, desired_class,
                                                       sample_custom_query_1, total_CFs,
                                                       version):
        counterfactual_explanations = self.exp.local_feature_importance(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)
        counterfactual_explanations.cf_examples_list[0].final_cfs_df = None
        counterfactual_explanations.cf_examples_list[0].final_cfs_df_sparse = None
        self.verify_counterfactual_explanations(counterfactual_explanations, None,
                                                sample_custom_query_1.shape[0], version,
                                                local_importance_available=True)
        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)
        self.verify_counterfactual_explanations(recovered_counterfactual_explanations, None,
                                                sample_custom_query_1.shape[0], version,
                                                local_importance_available=True)
        assert counterfactual_explanations == recovered_counterfactual_explanations

    @pytest.mark.parametrize("version", ['1.0', '2.0'])
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(0, 10)])
    def test_no_counterfactuals_found_summary_importance(self, desired_class,
                                                         sample_custom_query_10, total_CFs,
                                                         version):
        counterfactual_explanations = self.exp.global_feature_importance(
            query_instances=sample_custom_query_10, desired_class=desired_class,
            total_CFs=total_CFs)
        counterfactual_explanations.cf_examples_list[0].final_cfs_df = None
        counterfactual_explanations.cf_examples_list[0].final_cfs_df_sparse = None
        counterfactual_explanations.cf_examples_list[9].final_cfs_df = None
        counterfactual_explanations.cf_examples_list[9].final_cfs_df_sparse = None
        self.verify_counterfactual_explanations(counterfactual_explanations, None,
                                                sample_custom_query_10.shape[0], version,
                                                local_importance_available=True,
                                                summary_importance_available=True)
        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)
        self.verify_counterfactual_explanations(recovered_counterfactual_explanations, None,
                                                sample_custom_query_10.shape[0], version,
                                                local_importance_available=True,
                                                summary_importance_available=True)
        assert counterfactual_explanations == recovered_counterfactual_explanations

    @pytest.mark.parametrize('unsupported_version', ['3.0', ''])
    def test_unsupported_versions_from_json(self, unsupported_version):
        json_str = json.dumps({'metadata': {'version': unsupported_version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "Incompatible version {} found in json input".format(unsupported_version) in str(ucve)

        json_str = json.dumps({'metadata': {'versio': unsupported_version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "No version field in the json input" in str(ucve)

    @pytest.mark.parametrize('unsupported_version', ['3.0', ''])
    def test_unsupported_versions_to_json(self, unsupported_version):
        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance=None,
            version=unsupported_version)

        with pytest.raises(UserConfigValidationException) as ucve:
            counterfactual_explanations.to_json()

        assert "Unsupported serialization version {}".format(unsupported_version) in str(ucve)
