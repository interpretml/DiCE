import pytest
from collections import OrderedDict
import pandas as pd
import dice_ml
from dice_ml.utils import helpers

@pytest.fixture
def public_data_object():
    """
    Returns a public data object for the adult income dataset
    """
    dataset = helpers.load_adult_income_dataset()
    return dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')


@pytest.fixture
def private_data_object():
    """
    Returns a private data object containing meta information about the adult income dataset
    """
    features_dict = OrderedDict([('age', [17, 90]),
                            ('workclass', ['Government', 'Other/Unknown', 'Private', 'Self-Employed']),
                            ('education', ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college']),
                            ('marital_status', ['Divorced', 'Married', 'Separated', 'Single', 'Widowed']),
                            ('occupation', ['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar']),
                            ('race', ['Other', 'White']),
                            ('gender', ['Female', 'Male']),
                            ('hours_per_week', [1, 99])]) # providing an OrderedDict to make it work for Python<=3.6
    return dice_ml.Data(features=features_dict, outcome_name='income')

@pytest.fixture
def sample_adultincome_query():
    """
    Returns a sample query instance for adult income dataset
    """
    return {'age':22, 'workclass':'Private', 'education':'HS-grad', 'marital_status':'Single', 'occupation':'Service',
    'race': 'White', 'gender':'Female', 'hours_per_week': 45}

@pytest.fixture
def sample_custom_query_1():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['a'], 'Numerical': [25]})

@pytest.fixture
def sample_custom_query_2():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['b'], 'Numerical': [25]})

@pytest.fixture
def sample_custom_query_3():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['d'], 'Numerical': [1000000]})

@pytest.fixture
def sample_custom_query_4():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['c'], 'Numerical': [13]})
