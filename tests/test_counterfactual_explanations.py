import json
import pytest

from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.utils.exception import UserConfigValidationException


class TestCounterfactualExplanations:

    def test_serialization_deserialization_counterfactual_explanations_class(self):

        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance=None)
        assert counterfactual_explanations.cf_examples_list is not None
        assert len(counterfactual_explanations.cf_examples_list) == 0
        assert counterfactual_explanations.summary_importance is None
        assert counterfactual_explanations.local_importance is None
        assert counterfactual_explanations.metadata is not None
        assert counterfactual_explanations.metadata['version'] is not None
        assert counterfactual_explanations.metadata['version'] == '1.0'

        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)
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

    @pytest.mark.parametrize('version', ['2.0', ''])
    def test_unsupported_versions_json_input(self, version):
        json_str = json.dumps({'metadata': {'version': version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "Incompatible version {} found in json input".format(version) in str(ucve)

        json_str = json.dumps({'metadata': {'versio': version}})
        with pytest.raises(UserConfigValidationException) as ucve:
            CounterfactualExplanations.from_json(json_str)

        assert "No version field in the json input" in str(ucve)