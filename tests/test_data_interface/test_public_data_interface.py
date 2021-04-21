import numpy as np
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


class TestErrorCasesPublicDataInterface:
    def test_invalid_dataframe(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        with pytest.raises(ValueError) as ve:
            dice_ml.Data(dataframe=dataset.values, continuous_features=feature_names,
                         outcome_name='target')

        assert "should provide a pandas dataframe" in str(ve)

    def test_invalid_continuous_features(self):
        iris = load_iris(as_frame=True)
        feature_names = np.array(iris.feature_names)
        dataset = iris.frame

        with pytest.raises(ValueError) as ve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target')

        assert "should provide the name(s) of continuous features in the data as a list" in str(ve)

    def test_invalid_outcome_name(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        with pytest.raises(ValueError) as ve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name=1)

        assert "should provide the name of outcome feature as a string" in str(ve)

    def test_not_found_outcome_name(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='invalid')

        assert "outcome_name invalid not found in" in str(ucve)

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

    def test_unseen_continuous_features_precision(self):
        iris = load_iris(as_frame=True)
        feature_names = iris.feature_names
        continuous_features_precision = {'hours_per_week': 2}
        dataset = iris.frame

        with pytest.raises(UserConfigValidationException) as ucve:
            dice_ml.Data(dataframe=dataset, continuous_features=feature_names,
                         outcome_name='target',
                         continuous_features_precision=continuous_features_precision)

        assert "continuous_features_precision contains some feature names which are not part of columns in dataframe" in str(ucve)
