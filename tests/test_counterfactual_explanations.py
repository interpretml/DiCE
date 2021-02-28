import pytest

from dice_ml.counterfactual_explanations import CounterfactualExplanations


class TestCounterfactualExplanations:

    def test_counterfactual_explanations_class(self):
        counterfactual_explanations = CounterfactualExplanations(
            cf_examples_list=[],
            local_importance=None,
            summary_importance={
                "age": 0.985,
                "workclass": 0.65,
                "education": 0.915,
                "occupation": 0.95,
                "hours_per_week": 0.985,
                "gender": 0.67,
                "marital_status": 0.655,
                "race": 0.41})
        assert counterfactual_explanations.cf_examples_list is not None
        assert len(counterfactual_explanations.cf_examples_list) == 0
        assert counterfactual_explanations.summary_importance is not None
        assert counterfactual_explanations.local_importance is None
        assert counterfactual_explanations.metadata is not None
        assert counterfactual_explanations.metadata['version'] is not None
        assert counterfactual_explanations.metadata['version'] == '1.0'

        counterfactual_explanations_as_json = counterfactual_explanations.to_json()
        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations_as_json)
        assert counterfactual_explanations == recovered_counterfactual_explanations
