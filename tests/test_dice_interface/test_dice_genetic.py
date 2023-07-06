import pytest
import sklearn
from raiutils.exceptions import UserConfigValidationException

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.neuralnetworks import FFNetwork, MulticlassNetwork

BACKENDS = ['sklearn', 'PYT']


@pytest.fixture(scope="module", params=['sklearn'])
def genetic_binary_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    if backend == "PYT":
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max")
    else:
        ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_binary()
        m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


@pytest.fixture(scope="module", params=['sklearn'])
def genetic_multi_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_multiclass()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


@pytest.fixture(scope="module", params=BACKENDS)
def genetic_regression_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    if backend == "PYT":
        net = FFNetwork(4, is_classifier=False)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max", model_type='regressor')
    else:
        ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_regression()
        m = dice_ml.Model(model_path=ML_modelpath, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


class TestDiceGeneticBinaryClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, genetic_binary_classification_exp_object):
        self.exp = genetic_binary_classification_exp_object  # explainer object

    # When invalid desired_class is given
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(7, 3)])
    def test_no_cfs(self, desired_class, sample_custom_query_1, total_CFs):
        with pytest.raises(UserConfigValidationException):
            self.exp.generate_counterfactuals(query_instances=sample_custom_query_1, total_CFs=total_CFs,
                                              desired_class=desired_class)

    # When a query's feature value is not within the permitted range and the feature is not allowed to vary
    @pytest.mark.parametrize(("features_to_vary", "permitted_range", "feature_weights"),
                             [(['Numerical'], {'Categorical': ['b', 'c']}, "inverse_mad")])
    def test_invalid_query_instance(self, sample_custom_query_1, features_to_vary, permitted_range, feature_weights):
        with pytest.raises(
                ValueError,
                match="is outside the permitted range and isn't allowed to vary"):
            self.exp.setup(features_to_vary, permitted_range, sample_custom_query_1, feature_weights)

    # Testing that the features_to_vary argument actually varies only the features that you wish to vary
    @pytest.mark.parametrize(("desired_class", "total_CFs", "features_to_vary", "initialization"),
                             [(1, 2, ["Numerical"], "kdtree"), (1, 2, ["Numerical"], "random")])
    def test_features_to_vary(self, desired_class, sample_custom_query_2, total_CFs, features_to_vary, initialization):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                features_to_vary=features_to_vary,
                                                total_CFs=total_CFs, desired_class=desired_class,
                                                initialization=initialization)

        for cfs_example in ans.cf_examples_list:
            for feature in self.exp.data_interface.feature_names:
                if feature not in features_to_vary:
                    assert all(
                        cfs_example.final_cfs_df[feature].values[i] == sample_custom_query_2[feature].values[0] for i in
                        range(total_CFs))

    # Testing that the permitted_range argument actually varies the features only within the permitted_range
    @pytest.mark.parametrize(("desired_class", "total_CFs", "features_to_vary", "permitted_range", "initialization"),
                             [(1, 2, "all", {'Numerical': [10, 15]}, "kdtree"),
                              (1, 2, "all", {'Numerical': [10, 15]}, "random")])
    def test_permitted_range(self, desired_class, sample_custom_query_2, total_CFs, features_to_vary, permitted_range,
                             initialization):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                features_to_vary=features_to_vary, permitted_range=permitted_range,
                                                total_CFs=total_CFs, desired_class=desired_class,
                                                initialization=initialization)

        for cfs_example in ans.cf_examples_list:
            for feature in permitted_range:
                assert all(
                    permitted_range[feature][0] <= cfs_example.final_cfs_df[feature].values[i] <=
                    permitted_range[feature][1] for i
                    in range(total_CFs))

    # Testing if you can provide permitted_range for categorical variables
    @pytest.mark.parametrize(("desired_class", "total_CFs", "features_to_vary", "permitted_range", "initialization"),
                             [(1, 2, "all", {'Categorical': ['a', 'c']}, "kdtree"),
                              (1, 2, "all", {'Categorical': ['a', 'c']}, "random")])
    def test_permitted_range_categorical(self, desired_class, total_CFs, features_to_vary, sample_custom_query_2,
                                         permitted_range,
                                         initialization):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                features_to_vary=features_to_vary, permitted_range=permitted_range,
                                                total_CFs=total_CFs, desired_class=desired_class,
                                                initialization=initialization)

        for cfs_example in ans.cf_examples_list:
            for feature in permitted_range:
                assert all(
                    permitted_range[feature][0] <= cfs_example.final_cfs_df[feature].values[i] <=
                    permitted_range[feature][1] for i
                    in range(total_CFs))

    # Testing if an error is thrown when the query instance has outcome variable
    def test_query_instance_with_target_column(self, sample_custom_query_6):
        with pytest.raises(
                ValueError, match="present in query instance"):
            self.exp.setup("all", None, sample_custom_query_6, "inverse_mad")

    # Testing if only valid cfs are found after maxiterations
    @pytest.mark.parametrize(("desired_class", "total_CFs", "initialization", "maxiterations"),
                             [(0, 7, "kdtree", 0), (0, 7, "random", 0)])
    def test_maxiter(self, desired_class, sample_custom_query_2, total_CFs, initialization, maxiterations):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                total_CFs=total_CFs, desired_class=desired_class,
                                                initialization=initialization, maxiterations=maxiterations)
        for cfs_example in ans.cf_examples_list:
            for i in cfs_example.final_cfs_df[self.exp.data_interface.outcome_name].values:
                assert i == desired_class

    # Testing the custom predict function
    @pytest.mark.parametrize(("desired_class"), [2])
    def test_predict_custom(self, desired_class, sample_custom_query_2, mocker):
        self.exp.data_interface.set_continuous_feature_indexes(query_instance=sample_custom_query_2)
        self.exp.yloss_type = 'hinge_loss'
        mocker.patch('dice_ml.explainer_interfaces.dice_genetic.DiceGenetic.label_decode', return_value=None)
        mocker.patch('dice_ml.model_interfaces.base_model.BaseModel.get_output', return_value=[[0, 0.5, 0.5]])
        mocker.patch('dice_ml.model_interfaces.pytorch_model.PyTorchModel.get_output', return_value=[[0, 0.5, 0.5]])
        custom_preds = self.exp._predict_fn_custom(sample_custom_query_2, desired_class)
        assert custom_preds[0] == desired_class


class TestDiceGeneticMultiClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, genetic_multi_classification_exp_object):
        self.exp = genetic_multi_classification_exp_object  # explainer object

    # Testing if only valid cfs are found after maxiterations
    @pytest.mark.parametrize(("desired_class", "total_CFs", "initialization", "maxiterations"),
                             [(2, 7, "kdtree", 0), (2, 7, "random", 0)])
    def test_maxiter(self, desired_class, sample_custom_query_2, total_CFs, initialization, maxiterations):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                total_CFs=total_CFs, desired_class=desired_class,
                                                initialization=initialization, maxiterations=maxiterations)
        for cfs_example in ans.cf_examples_list:
            for i in cfs_example.final_cfs_df[self.exp.data_interface.outcome_name].values:
                assert i == desired_class

    # Testing the custom predict function
    @pytest.mark.parametrize(("desired_class"), [2])
    def test_predict_custom(self, desired_class, sample_custom_query_2, mocker):
        self.exp.data_interface.set_continuous_feature_indexes(query_instance=sample_custom_query_2)
        self.exp.yloss_type = 'hinge_loss'
        mocker.patch('dice_ml.explainer_interfaces.dice_genetic.DiceGenetic.label_decode', return_value=None)
        mocker.patch('dice_ml.model_interfaces.base_model.BaseModel.get_output', return_value=[[0, 0.5, 0.5]])
        custom_preds = self.exp._predict_fn_custom(sample_custom_query_2, desired_class)
        assert custom_preds[0] == desired_class

    # Testing if the shapes of the predictions are correct for multiclass classification
    @pytest.mark.parametrize(("desired_class", "method"), [(1, "genetic")])
    def test_multiclass_nn(self, desired_class, method):
        backend = "PYT"
        dataset = helpers.load_custom_testing_dataset_multiclass()

        # Transform the categorical data to numbers to test the neural network
        label_enc = sklearn.preprocessing.LabelEncoder()
        dataset['Categorical'] = label_enc.fit_transform(dataset['Categorical'])

        d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical', 'Categorical'], outcome_name='Outcome')

        # Load the neural network for multiclass classification and generate an explainer
        df = d.data_df
        num_class = len(df['Outcome'].unique())
        model = MulticlassNetwork(input_size=df.drop("Outcome", axis=1).shape[1], num_class=num_class)
        m = dice_ml.Model(model=model, backend=backend)
        exp = dice_ml.Dice(d, m, method=method)

        # Test the function that returns the predictions
        _, _, preds = exp.build_KD_tree(
            df.copy(), desired_range=None, desired_class=desired_class,
            predicted_outcome_name=d.outcome_name + '_pred'
        )
        assert hasattr(preds, "shape"), "The object that contains the predictions doesn't have a 'shape' attribute."
        assert preds.shape[0] == df.shape[0], "The number of predictions differs from the number of elements in the dataset."


class TestDiceGeneticRegressionMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, genetic_regression_exp_object):
        self.exp = genetic_regression_exp_object  # explainer object

    # features_range
    @pytest.mark.parametrize(("desired_range", "total_CFs", "initialization"),
                             [([1, 2.8], 2, "kdtree"), ([1, 2.8], 2, "random")])
    def test_desired_range(self, desired_range, sample_custom_query_2, total_CFs, initialization):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                total_CFs=total_CFs, desired_range=desired_range,
                                                initialization=initialization)
        for cfs_example in ans.cf_examples_list:
            assert all(
                [desired_range[0]] * total_CFs <= cfs_example.final_cfs_df[
                    self.exp.data_interface.outcome_name].values)
            assert all(
                cfs_example.final_cfs_df[self.exp.data_interface.outcome_name].values <= [desired_range[1]] * total_CFs)

    # Testing if only valid cfs are found after maxiterations
    @pytest.mark.parametrize(("desired_range", "total_CFs", "initialization", "maxiterations"),
                             [([1, 2.8], 7, "kdtree", 0), ([1, 2.8], 7, "random", 0)])
    def test_maxiter(self, desired_range, sample_custom_query_2, total_CFs, initialization, maxiterations):
        ans = self.exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                                total_CFs=total_CFs, desired_range=desired_range,
                                                initialization=initialization, maxiterations=maxiterations)
        for cfs_example in ans.cf_examples_list:
            for i in cfs_example.final_cfs_df[self.exp.data_interface.outcome_name].values:
                assert desired_range[0] <= i <= desired_range[1]
