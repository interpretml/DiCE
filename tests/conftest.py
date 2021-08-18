from collections import OrderedDict
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

import dice_ml
from dice_ml.utils import helpers


@pytest.fixture
def binary_classification_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_binary()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture
def binary_classification_exp_object_out_of_order(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_outcome_not_last_column_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_binary()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture
def multi_classification_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_multiclass()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture
def regression_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline_regression()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method=method)
    return exp


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
    Returns a private data object containing meta information about the adult income dataset.

    Providing an OrderedDict to make it work for Python<=3.6
    """
    features_dict = OrderedDict(
        [('age', [17, 90]),
         ('workclass', ['Government', 'Other/Unknown', 'Private', 'Self-Employed']),
         ('education', ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college']),
         ('marital_status', ['Divorced', 'Married', 'Separated', 'Single', 'Widowed']),
         ('occupation', ['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar']),
         ('race', ['Other', 'White']),
         ('gender', ['Female', 'Male']),
         ('hours_per_week', [1, 99])]
    )
    return dice_ml.Data(features=features_dict, outcome_name='income')


@pytest.fixture
def sample_adultincome_query():
    """
    Returns a sample query instance for adult income dataset
    """
    return {'age': 22, 'workclass': 'Private', 'education': 'HS-grad', 'marital_status': 'Single', 'occupation': 'Service',
            'race': 'White', 'gender': 'Female', 'hours_per_week': 45}


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


@pytest.fixture
def sample_custom_query_5():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'X': ['d'], 'Numerical': [25]})


@pytest.fixture
def sample_custom_query_6():
    """
    Returns a sample query instance for the custom dataset including Outcome
    """
    return pd.DataFrame({'Categorical': ['c'], 'Numerical': [13], 'Outcome': 0})


@pytest.fixture
def sample_custom_query_index():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['a'], 'Numerical': [88]})


@pytest.fixture
def sample_custom_query_10():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame(
        {
            'Categorical': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
            'Numerical': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        }
    )


@pytest.fixture
def sample_counterfactual_example_dummy():
    """
    Returns a sample counterfactual example
    """
    return pd.DataFrame(
        {
            'Categorical': ['a', 'b', 'c', 'a', 'b',
                            'c', 'a', 'b', 'c', 'a',
                            'a', 'b', 'c', 'c', 'c'],
            'Numerical': [25, 50, 75, 100, 125,
                          150, 175, 200, 225, 250,
                          150, 175, 200, 225, 250],
            'Outcome': [1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1]
        }
    )


@pytest.fixture
def create_iris_data():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)
    feature_names = iris.feature_names
    classes = iris.target_names
    return x_train, x_test, y_train, y_test, feature_names, classes


@pytest.fixture
def create_boston_data():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(
        boston.data, boston.target,
        test_size=0.2, random_state=7)
    x_train = pd.DataFrame(data=x_train, columns=boston.feature_names)
    x_test = pd.DataFrame(data=x_test, columns=boston.feature_names)
    return x_train, x_test, y_train, y_test, boston.feature_names.tolist()
