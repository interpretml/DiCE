import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import dice_ml
from dice_ml.utils.exception import UserConfigValidationException
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.diverse_counterfactuals import CounterfactualExamples


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseBinaryClassification:

    def _verify_feature_importance(self, feature_importance):
        if feature_importance is not None:
            for key in feature_importance:
                assert feature_importance[key] >= 0.0 and feature_importance[key] <= 1.0

    @pytest.mark.parametrize("desired_class", [1])
    def test_zero_totalcfs(
        self, desired_class, method, sample_custom_query_1,
        custom_public_data_interface_binary,
        sklearn_binary_classification_model_interface
    ):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)

        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)

    @pytest.mark.parametrize("desired_class", [1])
    def test_local_feature_importance(
            self, desired_class, method,
            sample_custom_query_1, sample_counterfactual_example_dummy,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
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

    @pytest.mark.parametrize("desired_class", [1])
    def test_global_feature_importance(
            self, desired_class, method,
            sample_custom_query_10, sample_counterfactual_example_dummy,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)

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

    @pytest.mark.parametrize("desired_class", [1])
    def test_columns_out_of_order(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_binary_out_of_order,
            sklearn_binary_classification_model_interface):
        if method == 'genetic':
            pytest.skip('DiceGenetic takes a very long time to run this test')

        exp = dice_ml.Dice(
            custom_public_data_interface_binary_out_of_order,
            sklearn_binary_classification_model_interface,
            method=method)

        cf_explanation = exp.generate_counterfactuals(
            query_instances=sample_custom_query_1,
            total_CFs=1,
            desired_class=desired_class,
            features_to_vary='all')

        assert cf_explanation is not None

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_features_to_vary_list(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
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

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_features_permitted_range(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
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

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_values_permitted_range(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
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

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize("desired_class", [100, 'a'])
    def test_unsupported_binary_class(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(query_instances=sample_custom_query_1, total_CFs=3,
                                         desired_class=desired_class)
        if desired_class == 100:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "The target class for {0} could not be identified".format(desired_class) in str(ucve)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_class", [1])
    def test_query_instance_unknown_column(
            self, desired_class, method, sample_custom_query_5,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError, match='not present in training data'):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_5, total_CFs=3,
                desired_class=desired_class)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_class", [1])
    def test_query_instance_outside_bounds(
            self, desired_class, method, sample_custom_query_3,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError, match='has a value outside the dataset'):
            exp.generate_counterfactuals(query_instances=sample_custom_query_3, total_CFs=1,
                                         desired_class=desired_class)

    # # Testing that the counterfactuals are in the desired class
    @pytest.mark.parametrize("desired_class", [1])
    def test_desired_class(
            self, desired_class, method, sample_custom_query_2,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        ans = exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                           features_to_vary='all',
                                           total_CFs=2, desired_class=desired_class,
                                           permitted_range=None)

        assert ans is not None
        assert len(ans.cf_examples_list) == sample_custom_query_2.shape[0]
        assert ans.cf_examples_list[0].final_cfs_df.shape[0] == 2

        if method != 'kdtree':
            assert all(ans.cf_examples_list[0].final_cfs_df[exp.data_interface.outcome_name].values == [desired_class] * 2)
        else:
            assert all(ans.cf_examples_list[0].final_cfs_df_sparse[exp.data_interface.outcome_name].values ==
                       [desired_class] * 2)

    @pytest.mark.parametrize("desired_class, total_CFs, permitted_range",
                             [(1, 1, {'Numerical': [10, 150]})])
    def test_permitted_range(
            self, desired_class, method, total_CFs, permitted_range, sample_custom_query_2,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        if method == 'kdtree':
            pytest.skip('DiceKD cannot seem to handle permitted_range')

        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        ans = exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                           permitted_range=permitted_range,
                                           total_CFs=total_CFs, desired_class=desired_class)

        for feature in permitted_range:
            if method != 'kdtree':
                assert all(
                    permitted_range[feature][0] <= ans.cf_examples_list[0].final_cfs_df[feature].values[i] <=
                    permitted_range[feature][1] for i in range(total_CFs))
            else:
                assert all(
                    permitted_range[feature][0] <= ans.cf_examples_list[0].final_cfs_df_sparse[feature].values[i] <=
                    permitted_range[feature][1] for i in range(total_CFs))

    # Testing for 0 CFs needed
    @pytest.mark.parametrize("features_to_vary, desired_class, desired_range, total_CFs, permitted_range",
                             [("all", 0, None, 0, None)])
    def test_zero_cfs_internal(
            self, method, features_to_vary, desired_class, desired_range, sample_custom_query_2, total_CFs,
            permitted_range, custom_public_data_interface_binary, sklearn_binary_classification_model_interface):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer does not handle the total counterfactuals as zero')
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        features_to_vary = exp.setup(features_to_vary, None, sample_custom_query_2, "inverse_mad")
        exp._generate_counterfactuals(features_to_vary=features_to_vary, query_instance=sample_custom_query_2,
                                      total_CFs=total_CFs, desired_class=desired_class,
                                      desired_range=desired_range, permitted_range=permitted_range)

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize("desired_class", [1])
    @pytest.mark.parametrize("permitted_range", [{'Categorical': ['b', 'c']}])
    @pytest.mark.parametrize("genetic_initialization", ['kdtree', 'random'])
    def test_permitted_range_categorical(
            self, method, desired_class, permitted_range, genetic_initialization,
            sample_custom_query_2, custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)

        if method != 'genetic':
            ans = exp.generate_counterfactuals(
                query_instances=sample_custom_query_2,
                permitted_range=permitted_range,
                total_CFs=2, desired_class=desired_class)
        else:
            ans = exp.generate_counterfactuals(
                query_instances=sample_custom_query_2,
                initialization=genetic_initialization,
                permitted_range=permitted_range,
                total_CFs=2, desired_class=desired_class)

        assert all(i in permitted_range["Categorical"] for i in ans.cf_examples_list[0].final_cfs_df.Categorical.values)

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize("desired_class, total_CFs, features_to_vary",
                             [(1, 1, ["Numerical"])])
    @pytest.mark.parametrize("genetic_initialization", ['kdtree', 'random'])
    def test_features_to_vary(
            self, method, desired_class, sample_custom_query_2,
            total_CFs, features_to_vary, genetic_initialization,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        if method != 'genetic':
            ans = exp.generate_counterfactuals(
                        query_instances=sample_custom_query_2,
                        features_to_vary=features_to_vary,
                        total_CFs=total_CFs, desired_class=desired_class)
        else:
            ans = exp.generate_counterfactuals(
                        query_instances=sample_custom_query_2,
                        features_to_vary=features_to_vary,
                        total_CFs=total_CFs, desired_class=desired_class,
                        initialization=genetic_initialization)

        for feature in exp.data_interface.feature_names:
            if feature not in features_to_vary:
                if method != 'kdtree':
                    assert all(
                        ans.cf_examples_list[0].final_cfs_df[feature].values[i] == sample_custom_query_2[feature].values[0]
                        for i in range(total_CFs))
                else:
                    assert all(
                        ans.cf_examples_list[0].final_cfs_df_sparse[feature].values[i] ==
                        sample_custom_query_2[feature].values[0]
                        for i in range(total_CFs))

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize("features_to_vary, permitted_range, feature_weights",
                             [(['Numerical'], {'Categorical': ['b', 'c']}, "inverse_mad")])
    def test_invalid_query_instance(
            self, method, sample_custom_query_1,
            features_to_vary, permitted_range,
            feature_weights,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError):
            exp.setup(features_to_vary, permitted_range, sample_custom_query_1, feature_weights)

    # Testing if an error is thrown when the query instance has outcome variable
    def test_query_instance_with_target_column(
            self, method, sample_custom_query_6,
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_binary,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError) as ve:
            exp.setup("all", None, sample_custom_query_6, "inverse_mad")

        assert "present in query instance" in str(ve)


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseMultiClassClassification:

    @pytest.mark.parametrize("desired_class", [1])
    def test_zero_totalcfs(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)

    # Testing that the counterfactuals are in the desired class
    @pytest.mark.parametrize("desired_class, total_CFs", [(2, 2)])
    @pytest.mark.parametrize("genetic_initialization", ['kdtree', 'random'])
    @pytest.mark.parametrize('posthoc_sparsity_algorithm', ['linear', 'binary', None])
    def test_desired_class(
            self, desired_class, total_CFs, method, genetic_initialization,
            posthoc_sparsity_algorithm,
            sample_custom_query_2,
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface,
            method=method)

        if method != 'genetic':
            ans = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class)
        else:
            ans = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class,
                    initialization=genetic_initialization,
                    posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)

        assert ans is not None
        if method != 'kdtree':
            assert all(
                ans.cf_examples_list[0].final_cfs_df[exp.data_interface.outcome_name].values == [desired_class] * total_CFs)
        else:
            assert all(
                ans.cf_examples_list[0].final_cfs_df_sparse[exp.data_interface.outcome_name].values ==
                [desired_class] * total_CFs)
        assert all(i == desired_class for i in exp.cfs_preds)

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize("desired_class, total_CFs", [(100, 3), ('opposite', 3)])
    def test_unsupported_multiclass(
            self, desired_class, total_CFs, method, sample_custom_query_4,
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(query_instances=sample_custom_query_4, total_CFs=total_CFs,
                                         desired_class=desired_class)
        if desired_class == 100:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "Desired class cannot be opposite if the number of classes is more than 2." in str(ucve)

    # Testing for 0 CFs needed
    @pytest.mark.parametrize("features_to_vary, desired_class, desired_range, total_CFs, permitted_range",
                             [("all", 0, None, 0, None)])
    def test_zero_cfs_internal(
            self, method, features_to_vary, desired_class, desired_range, sample_custom_query_2, total_CFs,
            permitted_range, custom_public_data_interface_multicalss, sklearn_multiclass_classification_model_interface):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer does not handle the total counterfactuals as zero')
        exp = dice_ml.Dice(
            custom_public_data_interface_multicalss,
            sklearn_multiclass_classification_model_interface,
            method=method)
        features_to_vary = exp.setup(features_to_vary, None, sample_custom_query_2, "inverse_mad")
        exp._generate_counterfactuals(features_to_vary=features_to_vary, query_instance=sample_custom_query_2,
                                      total_CFs=total_CFs, desired_class=desired_class,
                                      desired_range=desired_range, permitted_range=permitted_range)


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseRegression:

    @pytest.mark.parametrize("desired_range", [[10, 100]])
    def test_zero_cfs(
            self, desired_range, method,
            custom_public_data_interface_regression,
            sklearn_regression_model_interface,
            sample_custom_query_1):
        exp = dice_ml.Dice(
            custom_public_data_interface_regression,
            sklearn_regression_model_interface,
            method=method)

        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_range=desired_range)

    @pytest.mark.parametrize("desired_range", [[10, 100]])
    def test_numeric_categories(self, desired_range, method, create_boston_data):
        if method == 'genetic' or method == 'kdtree':
            pytest.skip('DiceGenetic/DiceKD explainer does not handle numeric categories')

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

    # Testing for 0 CFs needed
    @pytest.mark.parametrize("desired_range, total_CFs", [([1, 2.8], 0)])
    def test_zero_cfs_internal(
            self, desired_range, method, total_CFs,
            custom_public_data_interface_regression,
            sklearn_regression_model_interface,
            sample_custom_query_4):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer does not handle the total counterfactuals as zero')
        exp = dice_ml.Dice(
            custom_public_data_interface_regression,
            sklearn_regression_model_interface,
            method=method)

        exp._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                      desired_range=desired_range)

    @pytest.mark.parametrize("desired_range, total_CFs", [([1, 2.8], 6)])
    @pytest.mark.parametrize("version", ['2.0', '1.0'])
    @pytest.mark.parametrize('posthoc_sparsity_algorithm', ['linear', 'binary', None])
    def test_counterfactual_explanations_output(
            self, desired_range, total_CFs, method, version,
            posthoc_sparsity_algorithm, sample_custom_query_2,
            custom_public_data_interface_regression,
            sklearn_regression_model_interface):
        if method == 'genetic' and version == '1.0':
            pytest.skip('DiceGenetic cannot be serialized using version 1.0 serialization logic')

        exp = dice_ml.Dice(
            custom_public_data_interface_regression,
            sklearn_regression_model_interface,
            method=method)

        counterfactual_explanations = exp.generate_counterfactuals(
            query_instances=sample_custom_query_2, total_CFs=total_CFs,
            desired_range=desired_range,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)

        counterfactual_examples = counterfactual_explanations.cf_examples_list[0]

        if method != 'kdtree':
            assert all(
                [desired_range[0]] * counterfactual_examples.final_cfs_df.shape[0] <=
                counterfactual_examples.final_cfs_df[exp.data_interface.outcome_name].values) and \
                all(counterfactual_examples.final_cfs_df[exp.data_interface.outcome_name].values <=
                    [desired_range[1]] * counterfactual_examples.final_cfs_df.shape[0])
        else:
            assert all(
                [desired_range[0]] * counterfactual_examples.final_cfs_df_sparse.shape[0] <=
                counterfactual_examples.final_cfs_df_sparse[exp.data_interface.outcome_name].values) and \
                all(
                    counterfactual_examples.final_cfs_df_sparse[exp.data_interface.outcome_name].values <=
                    [desired_range[1]] * counterfactual_examples.final_cfs_df_sparse.shape[0])

        assert all(desired_range[0] <= i <= desired_range[1] for i in exp.cfs_preds)

        assert counterfactual_explanations is not None
        json_str = counterfactual_explanations.to_json()
        assert json_str is not None

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_str)
        assert recovered_counterfactual_explanations is not None
        assert counterfactual_explanations == recovered_counterfactual_explanations

        assert counterfactual_examples is not None
        json_str = counterfactual_examples.to_json(version)
        assert json_str is not None

        recovered_counterfactual_examples = CounterfactualExamples.from_json(json_str)
        assert recovered_counterfactual_examples is not None
        assert counterfactual_examples == recovered_counterfactual_examples


class TestExplainerBase:

    def test_instantiating_explainer_base(self, public_data_object):
        with pytest.raises(TypeError):
            ExplainerBase(data_interface=public_data_object)
