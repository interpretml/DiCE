import pytest

import dice_ml

@pytest.fixture
def data_object():
    return dice_ml.Data(features={'age':[17, 90],
                        'workclass': ['Government', 'Other/Unknown', 'Private', 'Self-Employed'],
                       'education': ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college'],
                       'marital_status': ['Divorced', 'Married', 'Separated', 'Single', 'Widowed'],
                       'occupation':['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar'],
                       'race': ['Other', 'White'],
                       'gender':['Female', 'Male'],
                       'hours_per_week': [1, 99]},
                    outcome_name='income',
                    type_and_precision={'hours_per_week':['float', 2]}, mad={'age': 10})

class TestPrivateDataMethods:
    @pytest.fixture(autouse=True)
    def _get_data_object(self, data_object):
        self.d = data_object

    def test_mads(self):
        mads = self.d.get_valid_mads(normalized=False)
        assert list(mads.values()) == [10.0, 1.0]

    def test_feature_precision(self):
        assert self.d.get_decimal_precisions()[1] == 2
