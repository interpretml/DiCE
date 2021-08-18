import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import dice_ml
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

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object_out_of_order",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['binary_classification_exp_object_out_of_order'])
    def test_columns_out_of_order(self, desired_class, binary_classification_exp_object_out_of_order, sample_custom_query_1):
        exp = binary_classification_exp_object_out_of_order  # explainer object
        exp._generate_counterfactuals(
            query_instance=sample_custom_query_1,
            total_CFs=0,
            desired_class=desired_class,
            desired_range=None,
            permitted_range=None,
            features_to_vary='all')

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['binary_classification_exp_object'])
    def test_incorrect_features_to_vary_list(self, desired_class, binary_classification_exp_object, sample_custom_query_1):
        exp = binary_classification_exp_object  # explainer object
        with pytest.raises(
                UserConfigValidationException,
                match="Got features {" + "'unknown_feature'" + "} which are not present in training data"):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range=None,
                features_to_vary=['unknown_feature'])

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['binary_classification_exp_object'])
    def test_incorrect_features_permitted_range(self, desired_class, binary_classification_exp_object, sample_custom_query_1):
        exp = binary_classification_exp_object  # explainer object
        with pytest.raises(
                UserConfigValidationException,
                match="Got features {" + "'unknown_feature'" + "} which are not present in training data"):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range={'unknown_feature': [1, 30]},
                features_to_vary='all')

    @pytest.mark.parametrize("desired_class, binary_classification_exp_object",
                             [(1, 'random'), (1, 'genetic'), (1, 'kdtree')],
                             indirect=['binary_classification_exp_object'])
    def test_incorrect_values_permitted_range(self, desired_class, binary_classification_exp_object, sample_custom_query_1):
        exp = binary_classification_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range={'Categorical': ['d']},
                features_to_vary='all')

        assert 'The category {0} does not occur in the training data for feature {1}. Allowed categories are {2}'.format(
            'd', 'Categorical', ['a', 'b', 'c']) in str(ucve)


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

    @pytest.mark.parametrize("desired_range, regression_exp_object",
                             [([10, 100], 'random'), ([10, 100], 'genetic'), ([10, 100], 'kdtree')],
                             indirect=['regression_exp_object'])
    def test_zero_totalcfs(self, desired_range, regression_exp_object, sample_custom_query_1):
        exp = regression_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_range=desired_range)

    @pytest.mark.parametrize("desired_range, method",
                             [([10, 100], 'random')])
    def test_numeric_categories(self, desired_range, method, create_boston_data):
        x_train, x_test, y_train, y_test, feature_names = \
            create_boston_data

        rfc = RandomForestRegressor(n_estimators=10, max_depth=4,
                                    random_state=777)
        model = rfc.fit(x_train, y_train)

        dataset_train = x_train.copy()
        dataset_train['Outcome'] = y_train
        feature_names.remove('CHAS')

        d = dice_ml.Data(dataframe=dataset_train, continuous_features=feature_names, outcome_name='Outcome')
        m = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')
        exp = dice_ml.Dice(d, m, method=method)

        cf_explanation = exp.generate_counterfactuals(
            query_instances=x_test.iloc[0:1],
            total_CFs=10,
            desired_range=desired_range)

        assert cf_explanation is not None


class TestExplainerBase:

    def test_instantiating_explainer_base(self, public_data_object):
        with pytest.raises(TypeError):
            ExplainerBase(data_interface=public_data_object)
