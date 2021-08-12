from enum import Enum
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException


@pytest.fixture
def data_object():
    dataset = helpers.load_adult_income_dataset()
    return dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'],
                        outcome_name='income', permitted_range={'age': [45, 60]},
                        continuous_features_precision={'hours_per_week': 2})


class TestPublicDataMethods:
    @pytest.fixture(autouse=True)
    def _get_data_object(self, data_object):
        self.d = data_object

    def test_permitted_range(self):
        self.d.create_ohe_params()
        minx, maxx = self.d.get_minx_maxx(normalized=False)
        assert [minx[0][0], maxx[0][0]] == [45, 60]
        minx, maxx = self.d.get_minx_maxx(normalized=True)
        assert pytest.approx([minx[0][0], maxx[0][0]], abs=1e-3) == [0.384, 0.589]

    def test_feature_precision(self):
        # feature precision decides the least change that can be made to the feature in
        # optimization, given as 2-decimal place for 'hours_per_week' feature while initiating
        # private Data object.
        assert self.d.get_decimal_precisions()[1] == 2


class DataTypeCombinations(Enum):
    Incorrect = 0
    AsNone = 1
    Omitted = 2


class TestErrorScenariosPublicDataInterface:
    @pytest.mark.parametrize('data_type', [DataTypeCombinations.Incorrect,
                                           DataTypeCombinations.AsNone])
    def test_invalid_dataframe(self, data_type):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        if data_type == DataTypeCombinations.Incorrect:
            with pytest.raises(ValueError, match="should provide a pandas dataframe"):
                dice_ml.Data(dataframe=dataset.values, continuous_features=feature_names,
                             outcome_name='target')
        else:
            with pytest.raises(ValueError, match="should provide a pandas dataframe"):
                dice_ml.Data(dataframe=None, continuous_features=feature_names,
                             outcome_name='target')

    @pytest.mark.parametrize('data_type', [DataTypeCombinations.Incorrect,
                                           DataTypeCombinations.AsNone,
                                           DataTypeCombinations.Omitted])
    def test_invalid_continuous_features(self, data_type):
        iris = load_iris(as_frame=True)
        dataset = iris.frame

        if data_type == DataTypeCombinations.Incorrect:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, continuous_features=np.array(iris.feature_names),
                             outcome_name='target')
            assert "should provide the name(s) of continuous features in the data as a list" in str(ve)
        elif data_type == DataTypeCombinations.AsNone:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, continuous_features=None,
                             outcome_name='target')
            assert "should provide the name(s) of continuous features in the data as a list" in str(ve)
        else:
            with pytest.raises(
                    ValueError,
                    match='continuous_features should be provided'):
                dice_ml.Data(dataframe=dataset, outcome_name='target')

    @pytest.mark.parametrize('data_type', [DataTypeCombinations.Incorrect,
                                           DataTypeCombinations.AsNone,
                                           DataTypeCombinations.Omitted])
    def test_invalid_outcome_name(self, data_type):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        if data_type == DataTypeCombinations.Incorrect:
            with pytest.raises(
                    ValueError,
                    match="should provide the name of outcome feature as a string"):
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                             outcome_name=1)
        elif data_type == DataTypeCombinations.AsNone:
            with pytest.raises(
                    ValueError,
                    match="should provide the name of outcome feature as a string"):
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                             outcome_name=None)
        else:
            with pytest.raises(
                    ValueError,
                    match="should provide the name of outcome feature"):
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names)

    def test_not_found_outcome_name(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        with pytest.raises(
                UserConfigValidationException,
                match="outcome_name invalid not found in"):
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='invalid')

    def test_unseen_continuous_feature_names(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        feature_names.append("new feature")
        dataset = iris.frame

        with pytest.raises(
                UserConfigValidationException,
                match="continuous_features contains some feature names which are not part of columns in dataframe"):
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target')

    def test_unseen_permitted_range(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        permitted_range = {'age': [45, 60]}
        dataset = iris.frame

        with pytest.raises(
                UserConfigValidationException,
                match="permitted_range contains some feature names which are not part of columns in dataframe"):
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target', permitted_range=permitted_range)

    def test_min_max_equal(self):
        dataset = helpers.load_min_max_equal_dataset()
        dice_data = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
        assert all(dice_data.normalize_data(dice_data.data_df)['Numerical'] == 0)

    def test_unseen_continuous_features_precision(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        continuous_features_precision = {'hours_per_week': 2}
        dataset = iris.frame

        with pytest.raises(
                UserConfigValidationException,
                match="continuous_features_precision contains some feature names which"
                      " are not part of columns in dataframe"):
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target',
                         continuous_features_precision=continuous_features_precision)


class TestChecksPublicDataInterface:
    @pytest.mark.parametrize('features_to_vary', ['all', None, ['not_a_feature']])
    def test_check_features_to_vary(self, features_to_vary):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        dice_data = dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                                 outcome_name='target')

        if features_to_vary is not None and features_to_vary != 'all':
            with pytest.raises(
                    UserConfigValidationException,
                    match="Got features {" + "'not_a_feature'" + "} which are not present in training data"):
                dice_data.check_features_to_vary(features_to_vary=features_to_vary)
        else:
            dice_data.check_features_to_vary(features_to_vary=features_to_vary)

    @pytest.mark.parametrize('permitted_range', [None, {'not_a_feature': [20, 30]}])
    def test_check_permitted_range_with_unknown_feature(self, permitted_range):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        dice_data = dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                                 outcome_name='target')

        if permitted_range is not None:
            with pytest.raises(
                    UserConfigValidationException,
                    match="Got features {" + "'not_a_feature'" + "} which are not present in training data"):
                dice_data.check_permitted_range(permitted_range=permitted_range)
        else:
            dice_data.check_permitted_range(permitted_range=permitted_range)

    def test_check_permitted_range_with_unknown_categorical_value(self):
        iris = load_iris(as_frame=True)
        permitted_range = {'new_feature': ['unknown_category']}
        feature_names = iris.feature_names
        dataset = iris.frame
        dataset['new_feature'] = np.repeat(['known_category'], dataset.shape[0])

        dice_data = dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                                 outcome_name='target')

        with pytest.raises(
                UserConfigValidationException) as ucve:
            dice_data.check_permitted_range(permitted_range=permitted_range)

        assert 'The category {0} does not occur in the training data for feature {1}. Allowed categories are {2}'.format(
            'unknown_category', 'new_feature', ['known_category']) in str(ucve)


class TestUserDataCorruption:

    def test_user_data_corruption(self):
        dataset = helpers.load_adult_income_dataset()
        dataset_copy = dataset.copy()
        dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'],
                     outcome_name='income', permitted_range={'age': [45, 60]},
                     continuous_features_precision={'hours_per_week': 2})
        pd.testing.assert_frame_equal(dataset, dataset_copy)
