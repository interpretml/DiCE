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
    Incorrect = 1
    AsNone = 2
    Omitted = 2


class TestErrorCasesPublicDataInterface:
    @pytest.mark.parametrize('data_type', [DataTypeCombinations.Incorrect,
                                           DataTypeCombinations.AsNone,
                                           DataTypeCombinations.Omitted])
    def test_invalid_dataframe(self, data_type):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        if data_type == DataTypeCombinations.Incorrect:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset.values, continuous_features=feature_names,
                             outcome_name='target')
            assert "should provide a pandas dataframe" in str(ve)
        elif data_type == DataTypeCombinations.AsNone:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=None, continuous_features=feature_names,
                             outcome_name='target')
            assert "should provide a pandas dataframe" in str(ve)
        else:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(continuous_features=feature_names,
                             outcome_name='target')
            assert "dataframe not found in params" in str(ve)

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
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, outcome_name='target')

            assert 'continuous_features should be provided' in str(ve)

    @pytest.mark.parametrize('data_type', [DataTypeCombinations.Incorrect,
                                           DataTypeCombinations.AsNone,
                                           DataTypeCombinations.Omitted])
    def test_invalid_outcome_name(self, data_type):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        if data_type == DataTypeCombinations.Incorrect:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                             outcome_name=1)

            assert "should provide the name of outcome feature as a string" in str(ve)
        elif data_type == DataTypeCombinations.AsNone:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                             outcome_name=None)

            assert "should provide the name of outcome feature as a string" in str(ve)
        else:
            with pytest.raises(ValueError) as ve:
                dice_ml.Data(dataframe=dataset, continuous_features=feature_names)

            assert "should provide the name of outcome feature" in str(ve)

    def test_not_found_outcome_name(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='invalid')

        assert "outcome_name invalid not found in" in str(ucve)

    def test_outcome_not_last_column(self):
        dataset = helpers.load_outcome_not_last_column_dataset()
        with pytest.raises(ValueError) as ve:
            dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'],
                         outcome_name='Outcome')
        assert "Outcome should be the last column! Please reorder!" in str(ve)

    def test_unseen_continuous_feature_names(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        feature_names.append("new feature")
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target')

        assert "continuous_features contains some feature names which are not part of columns in dataframe" in str(ucve)

    def test_unseen_permitted_range(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        permitted_range = {'age': [45, 60]}
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target', permitted_range=permitted_range)

        assert "permitted_range contains some feature names which are not part of columns in dataframe" in str(ucve)

    def test_min_max_equal(self):
        dataset = helpers.load_min_max_equal_dataset()
        dice_data = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
        assert all(dice_data.normalize_data(dice_data.data_df)['Numerical'] == 0)

    def test_unseen_continuous_features_precision(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        continuous_features_precision = {'hours_per_week': 2}
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target',
                         continuous_features_precision=continuous_features_precision)

        assert "continuous_features_precision contains some feature names which" + \
            " are not part of columns in dataframe" in str(ucve)


class TestUserDataCorruption:

    def test_user_data_corruption(self):
        dataset = helpers.load_adult_income_dataset()
        dataset_copy = dataset.copy()
        dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'],
                     outcome_name='income', permitted_range={'age': [45, 60]},
                     continuous_features_precision={'hours_per_week': 2})
        pd.testing.assert_frame_equal(dataset, dataset_copy)
