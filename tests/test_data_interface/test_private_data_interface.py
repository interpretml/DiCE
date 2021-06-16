import pytest
from collections import OrderedDict

import dice_ml


@pytest.fixture
def data_object():
    features_dict = OrderedDict(
        [('age', [17, 90]),
         ('workclass', ['Government', 'Other/Unknown', 'Private', 'Self-Employed']),
         ('education', ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college']),
         ('marital_status', ['Divorced', 'Married', 'Separated', 'Single', 'Widowed']),
         ('occupation', ['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar']),
         ('race', ['Other', 'White']),
         ('gender', ['Female', 'Male']),
         ('hours_per_week', [1, 99])]
    )  # providing an OrderedDict to make it work for Python<3.6
    return dice_ml.Data(features=features_dict, outcome_name='income',
                        type_and_precision={'hours_per_week': ['float', 2]}, mad={'age': 10})


class TestPrivateDataMethods:
    @pytest.fixture(autouse=True)
    def _get_data_object(self, data_object):
        self.d = data_object

    def test_mads(self):
        # normalized=True is already tested in test_data.py
        mads = self.d.get_valid_mads(normalized=False)
        # 10 is given as the mad of feature 'age' while initiating private Data object; 1.0 is the default value.
        # Check get_valid_mads() in private_data_interface for more info.
        assert list(mads.values()) == [10.0, 1.0]

    def test_feature_precision(self):
        # feature precision decides the least change that can be made to the feature in optimization,
        # given as 2-decimal place for 'hours_per_week' feature while initiating private Data object.
        assert self.d.get_decimal_precisions()[1] == 2
