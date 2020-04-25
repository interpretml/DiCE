import pytest

import dice_ml
from dice_ml.utils import helpers

@pytest.fixture
def data_object():
    dataset = helpers.load_adult_income_dataset()
    d_public = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

    d_private = dice_ml.Data(features={'age':[17, 90],
                        'workclass': ['Government', 'Other/Unknown', 'Private', 'Self-Employed'],
                       'education': ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college'],
                       'marital_status': ['Divorced', 'Married', 'Separated', 'Single', 'Widowed'],
                       'occupation':['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar'],
                       'race': ['Other', 'White'],
                       'gender':['Female', 'Male'],
                       'hours_per_week': [1, 99]},
             outcome_name='income')
    return d_public, d_private

def test_data_initiation(data_object):
    assert isinstance(data_object[0], dice_ml.data_interfaces.public_data_interface.PublicData), "the given parameters should instantiate PublicData class"

    assert isinstance(data_object[1], dice_ml.data_interfaces.private_data_interface.PrivateData), "the given parameters should instantiate PrivateData class"

class TestCommonDataMethods:
    @pytest.fixture(autouse=True)
    def _get_data_object(self, data_object):
        self.d = data_object

    def test_get_valid_mads(self):
        # public data
        mads = self.d[0].get_valid_mads(normalized=False, display_warnings=False, return_mads=True)
        assert all(mads[feature] > 0 for feature in mads)

        mads = self.d[0].get_valid_mads(normalized=True, display_warnings=False, return_mads=True)
        errors = 0
        for feature in mads:
            if((mads[feature] < 0) | (mads[feature] > 1)):
                errors += 1
        assert errors==0

        # private data
        for normalized in [True, False]:
            mads = self.d[1].get_valid_mads(normalized=normalized, display_warnings=False, return_mads=True)
            assert all(mads[feature] == 1 for feature in mads)

    def test_prepare_query_instance(self):
        test_query_list = [0.068, 'Private', 'HS-grad', 'Single', 'Service', 'White', 'Female', 0.449]
        query_instance = {'age':22,
                  'workclass':'Private',
                  'education':'HS-grad',
                  'marital_status':'Single',
                  'occupation':'Service',
                  'race': 'White',
                  'gender':'Female',
                  'hours_per_week': 45}

        for d in self.d:
            prepared_query = d.prepare_query_instance(query_instance=query_instance, encode=False).iloc[0].tolist()
            prepared_query[0] = round(prepared_query[0], 3)
            prepared_query[-1] = round(prepared_query[-1], 3)
            assert test_query_list == prepared_query

    def test_encoded_categorical_features(self):
        res = []
        for d in self.d:
            d.categorical_feature_names = ['cat1', 'cat2']
            d.encoded_feature_names = ['cat2_cont1', 'cont2', 'cat1_val1', 'cat1_val2', 'cat2_val1', 'cat2_val2']
            d.continuous_feature_names = ['cat2_cont1', 'cont2']
            res.append(d.get_encoded_categorical_feature_indexes())
        assert [[2, 3], [4, 5]] == res[0]
        assert res[0] == res[1]

    def test_features_to_vary(self):
        res = []
        for d in self.d:
            d.categorical_feature_names = ['cat1', 'cat2']
            d.encoded_feature_names = ['cat2_cont1', 'cont2', 'cat1_val1', 'cat1_val2', 'cat2_val1', 'cat2_val2']
            d.continuous_feature_names = ['cat2_cont1', 'cont2']
            res.append(d.get_indexes_of_features_to_vary(features_to_vary=['cat2']))
        assert [4, 5] == res[0]
        assert res[0] == res[1]
