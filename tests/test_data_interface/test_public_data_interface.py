import pytest

import dice_ml
from dice_ml.utils import helpers

@pytest.fixture
def data_object():
    dataset = helpers.load_adult_income_dataset()
    return dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income', permitted_range={'age':[45,60]}, continuous_features_precision={'hours_per_week': 2})

class TestPublicDataMethods:
    @pytest.fixture(autouse=True)
    def _get_data_object(self, data_object):
        self.d = data_object

    def test_permitted_range(self):
        self.d.create_ohe_params()
        minx, maxx = self.d.get_minx_maxx(normalized=False)
        assert [minx[0][0], maxx[0][0]] == [45,60]
        minx, maxx = self.d.get_minx_maxx(normalized=True)
        assert pytest.approx([minx[0][0], maxx[0][0]], abs=1e-3) == [0.384, 0.589]

    def test_feature_precision(self):
        assert self.d.get_decimal_precisions()[1] == 2 # feature precision decides the least change that can be made to the feature in optimization, given as 2-decimal place for 'hours_per_week' feature while initiating private Data object.
