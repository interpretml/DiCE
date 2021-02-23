import pytest

from dice_ml.counterfactual_explanations import CounterfactualExplanations


class TestCounterfactualExplanations:

    def test_counterfactual_explanations_class(self):
        counterfactual_explanations = CounterfactualExplanations(cf_examples_list=[])
        assert counterfactual_explanations.cf_examples_list is not None
        assert len(counterfactual_explanations.cf_examples_list) == 0
        assert counterfactual_explanations.summary_importance is None
        assert counterfactual_explanations.local_importance is None
        assert counterfactual_explanations.metadata is not None
        assert counterfactual_explanations.metadata['version'] is not None
        assert counterfactual_explanations.metadata['version'] == '1.0'
