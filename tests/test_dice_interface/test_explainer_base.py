import pytest
from dice_ml.utils.exception import UserConfigValidationException


class TestExplainerBaseBinaryClassification:

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object", [(1, 'random'),(1,'genetic'),(1,'kdtree')], indirect=['binary_classification_exp_object'])
    def test_zero_totalcfs(self, desired_class, binary_classification_exp_object, sample_custom_query_1):
        exp = binary_classification_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)

class TestExplainerBaseMultiClassClassification:

    @pytest.mark.parametrize("desired_class, multi_classification_exp_object", [(1, 'random'),(1,'genetic'),(1,'kdtree')], indirect=['multi_classification_exp_object'])
    def test_zero_totalcfs(self, desired_class, multi_classification_exp_object, sample_custom_query_1):
        exp = multi_classification_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)


class TestExplainerBaseRegression:

    @pytest.mark.parametrize("desired_class, regression_exp_object", [(1, 'random'),(1,'genetic'),(1,'kdtree')], indirect=['regression_exp_object'])
    def test_zero_totalcfs(self, desired_class, regression_exp_object, sample_custom_query_1):
        exp = regression_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)
