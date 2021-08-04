import pandas as pd
import pytest

from dice_ml.utils.exception import UserConfigValidationException
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class TestExplainerBaseBinaryClassification:

    def _verify_feature_importance(self, feature_importance):
        if feature_importance is not None:
            for key in feature_importance:
                assert feature_importance[key] >= 0.0 and feature_importance[key] <= 1.0

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['binary_classification_exp_object'])
    def test_zero_totalcfs(self, desired_class, binary_classification_exp_object, sample_custom_query_1):
        exp = binary_classification_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random')],
                             indirect=['binary_classification_exp_object'])
    def test_local_feature_importance(self, desired_class, binary_classification_exp_object,
                                      sample_custom_query_1, sample_counterfactual_example_dummy):
        exp = binary_classification_exp_object  # explainer object
        sample_custom_query = pd.concat([sample_custom_query_1, sample_custom_query_1])
        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query,
                    total_CFs=15,
                    desired_class=desired_class)

        cf_explanations.cf_examples_list[0].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df.drop([0, 1, 2], inplace=True)
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse.drop([0, 1, 2], inplace=True)

        cf_explanations.cf_examples_list[1].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[1].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[1].final_cfs_df.drop([0], inplace=True)
        cf_explanations.cf_examples_list[1].final_cfs_df_sparse.drop([0], inplace=True)

        local_importances = exp.local_feature_importance(
            query_instances=None,
            cf_examples_list=cf_explanations.cf_examples_list)

        for local_importance in local_importances.local_importance:
            self._verify_feature_importance(local_importance)


    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random')],
                             indirect=['binary_classification_exp_object'])
    def test_global_feature_importance(self, desired_class, binary_classification_exp_object,
                                       sample_custom_query_10, sample_counterfactual_example_dummy):
        exp = binary_classification_exp_object  # explainer object
        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_10,
                    total_CFs=15,
                    desired_class=desired_class)

        cf_explanations.cf_examples_list[0].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df.drop([0, 1, 2, 3, 4], inplace=True)
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse.drop([0, 1, 2, 3, 4], inplace=True)

        for index in range(1, len(cf_explanations.cf_examples_list)):
            cf_explanations.cf_examples_list[index].final_cfs_df = sample_counterfactual_example_dummy.copy()
            cf_explanations.cf_examples_list[index].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()

        global_importance = exp.global_feature_importance(
            query_instances=None,
            cf_examples_list=cf_explanations.cf_examples_list)

        self._verify_feature_importance(global_importance.summary_importance)


class TestExplainerBaseMultiClassClassification:

    @pytest.mark.parametrize("desired_class, multi_classification_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['multi_classification_exp_object'])
    def test_zero_totalcfs(self, desired_class, multi_classification_exp_object, sample_custom_query_1):
        exp = multi_classification_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)


class TestExplainerBaseRegression:

    @pytest.mark.parametrize("desired_class, regression_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['regression_exp_object'])
    def test_zero_totalcfs(self, desired_class, regression_exp_object, sample_custom_query_1):
        exp = regression_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)


class TestExplainerBase:

    def test_instantiating_explainer_base(self, public_data_object):
        with pytest.raises(TypeError):
            ExplainerBase(data_interface=public_data_object)
