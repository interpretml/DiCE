from collections import OrderedDict
from itertools import product

import pandas as pd
import pytest
import torch
from rai_test_utils.models.sklearn import (
    create_complex_classification_pipeline, create_complex_regression_pipeline)

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.neuralnetworks import FFNetwork

BACKENDS = ['sklearn', 'PYT']

DATA_INTERFACES = ['private', 'public']


@pytest.fixture(scope="session", params=product(BACKENDS, DATA_INTERFACES))
def random_binary_classification_exp_object(request):
    backend, dinterface = request.param
    if dinterface == "public":
        dataset = helpers.load_custom_testing_dataset_binary()
        d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    else:
        d = dice_ml.Data(features={
                                   'Numerical': [0, 10],
                                   'Categorical': ['a', 'b', 'c',]},
                         outcome_name="Outcome")
    if backend == "PYT":
        torch.manual_seed(1)
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max")
    else:
        model = _load_custom_testing_binary_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


@pytest.fixture(scope="session", params=product(["sklearn"], DATA_INTERFACES))
def random_str_binary_classification_exp_object(request):
    backend, dinterface = request.param
    if dinterface == "public":
        dataset = helpers.load_custom_testing_dataset_binary_str()
        d = dice_ml.Data(
            dataframe=dataset, continuous_features=["Numerical"], outcome_name="Outcome"
        )
    else:
        d = dice_ml.Data(
            features={"Numerical": [0, 5], "Categorical": ["a", "b", "c"]},
            outcome_name="Outcome",
        )
    if backend == "PYT":
        torch.manual_seed(1)
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend, func="ohe-min-max")
    else:
        model = _load_custom_testing_binary_str_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method="random")
    return exp


# TODO multiclass is not currently supported for neural networks
@pytest.fixture(scope="module", params=product(['sklearn'], DATA_INTERFACES))
def random_multi_classification_exp_object(request):
    backend, dinterface = request.param
    if dinterface == "public":
        dataset = helpers.load_custom_testing_dataset_multiclass()
        d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    else:
        d = dice_ml.Data(features={
                                   'Numerical': [7, 23],
                                   'Categorical': ['a', 'b', 'c']},
                         outcome_name="Outcome")
    if backend == "PYT":
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max")
    else:
        model = _load_custom_testing_multiclass_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


@pytest.fixture(scope="module", params=product(['sklearn'], DATA_INTERFACES))
def random_str_multi_classification_exp_object(request):
    backend, dinterface = request.param
    if dinterface == "public":
        dataset = helpers.load_custom_testing_dataset_multiclass_str()
        d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    else:
        d = dice_ml.Data(features={
                                   'Numerical': [7, 23],
                                   'Categorical': ['a', 'b', 'c']},
                         outcome_name="Outcome")
    if backend == "PYT":
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max")
    else:
        model = _load_custom_testing_multiclass_str_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


@pytest.fixture(scope="module", params=product(BACKENDS, DATA_INTERFACES))
def random_regression_exp_object(request):
    backend, dinterface = request.param
    if dinterface == 'public':
        dataset = helpers.load_custom_testing_dataset_regression()
        d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    else:
        d = dice_ml.Data(features={
                                   'Numerical': [7, 23],
                                   'Categorical': ['a', 'b', 'c']},
                         outcome_name="Outcome")
    if backend == "PYT":
        net = FFNetwork(4, is_classifier=False)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max", model_type='regressor')
    else:
        model = _load_custom_testing_regression_model()
        m = dice_ml.Model(model=model, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method='random')
    return exp


@pytest.fixture(scope="module", params=['sklearn'])
def genetic_binary_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    if backend == "PYT":
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend,  func="ohe-min-max")
    else:
        model = _load_custom_testing_binary_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


@pytest.fixture(scope="module", params=["sklearn"])
def genetic_binary_str_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_binary_str()
    d = dice_ml.Data(
        dataframe=dataset, continuous_features=["Numerical"], outcome_name="Outcome"
    )
    if backend == "PYT":
        net = FFNetwork(4)
        m = dice_ml.Model(model=net, backend=backend, func="ohe-min-max")
    else:
        model = _load_custom_testing_binary_str_model()
        m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method="genetic")
    return exp


@pytest.fixture(scope="module", params=['sklearn'])
def genetic_multi_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_multiclass_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


@pytest.fixture(scope="module", params=['sklearn'])
def genetic_str_multi_classification_exp_object(request):
    backend = request.param
    dataset = helpers.load_custom_testing_dataset_multiclass_str()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_multiclass_str_model()
    m = dice_ml.Model(model=model, backend=backend)
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
        model = _load_custom_testing_regression_model()
        m = dice_ml.Model(model=model, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method='genetic')
    return exp


@pytest.fixture(scope='session')
def KD_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_binary_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
def KD_binary_vars_classification_exp_object():
    backend = 'sklearn'
    dataset = load_custom_vars_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_vars_dataset_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
def KD_multi_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_multiclass_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope='session')
def KD_regression_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_regression_model()
    m = dice_ml.Model(model=model, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


@pytest.fixture(scope="session")
def binary_classification_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_binary()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_binary_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture(scope="session")
def multi_classification_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_multiclass()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_multiclass_model()
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture(scope="session")
def regression_exp_object(method="random"):
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    model = _load_custom_testing_regression_model()
    m = dice_ml.Model(model=model, backend=backend, model_type='regressor')
    exp = dice_ml.Dice(d, m, method=method)
    return exp


@pytest.fixture(scope='session')
def custom_public_data_interface():
    dataset = helpers.load_custom_testing_dataset_regression()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    return d


@pytest.fixture(scope='session')
def sklearn_binary_classification_model_interface():
    model = _load_custom_testing_binary_model()
    m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')
    return m


@pytest.fixture(scope='session')
def sklearn_multiclass_classification_model_interface():
    model = _load_custom_testing_multiclass_model()
    m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')
    return m


@pytest.fixture(scope='session')
def sklearn_regression_model_interface():
    model = _load_custom_testing_regression_model()
    m = dice_ml.Model(model=model, backend='sklearn', model_type='regression')
    return m


@pytest.fixture(scope='session')
def public_data_object():
    """
    Returns a public data object for the adult income dataset
    """
    dataset = helpers.load_adult_income_dataset()
    return dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')


@pytest.fixture(scope='session')
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


def _load_custom_testing_model():
    dataset = helpers.load_custom_testing_dataset()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def _load_custom_testing_binary_model():
    dataset = helpers.load_custom_testing_dataset_binary()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def _load_custom_testing_binary_str_model():
    dataset = helpers.load_custom_testing_dataset_binary_str()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def _load_custom_testing_multiclass_model():
    dataset = helpers.load_custom_testing_dataset_multiclass()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def _load_custom_testing_multiclass_str_model():
    dataset = helpers.load_custom_testing_dataset_multiclass_str()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def _load_custom_testing_regression_model():
    dataset = helpers.load_custom_testing_dataset_regression()
    X_train = dataset[["Categorical", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical"]
    model = create_complex_regression_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


def load_custom_vars_testing_dataset():
    data = [['a', 0, 10, 0], ['b', 1, 10000, 1], ['c', 0, 14, 0], ['a', 2, 88, 0], ['c', 1, 14, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'CategoricalNum', 'Numerical', 'Outcome'])


def _load_custom_vars_dataset_model():
    dataset = load_custom_vars_testing_dataset()
    X_train = dataset[["Categorical", "CategoricalNum", "Numerical"]]
    y_train = dataset["Outcome"].values
    num_feature_names = ["Numerical"]
    cat_feature_names = ["Categorical", "CategoricalNum"]
    model = create_complex_classification_pipeline(
        X_train, y_train, num_feature_names, cat_feature_names)
    return model


@pytest.fixture(scope='session')
def sample_adultincome_query():
    """
    Returns a sample query instance for adult income dataset
    """
    return pd.DataFrame({
        'age': 22, 'workclass': 'Private', 'education': 'HS-grad',
        'marital_status': 'Single', 'occupation': 'Service',
        'race': 'White', 'gender': 'Female', 'hours_per_week': 45},
        index=[0])


@pytest.fixture(scope='session')
def sample_custom_query_1():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['a'], 'Numerical': [25]})


@pytest.fixture(scope='session')
def sample_custom_query_2():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['b'], 'Numerical': [25]})


@pytest.fixture(scope='session')
def sample_custom_query_3():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['d'], 'Numerical': [1000000]})


@pytest.fixture(scope='session')
def sample_custom_query_4():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['c'], 'Numerical': [13]})


@pytest.fixture(scope='session')
def sample_custom_query_5():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'X': ['d'], 'Numerical': [25]})


@pytest.fixture(scope='session')
def sample_custom_query_6():
    """
    Returns a sample query instance for the custom dataset including Outcome
    """
    return pd.DataFrame({'Categorical': ['c'], 'Numerical': [13], 'Outcome': 0})


@pytest.fixture(scope='session')
def sample_custom_query_index():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['a'], 'Numerical': [88]})


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def sample_custom_vars_query_1():
    """
    Returns a sample query instance for the custom dataset
    """
    return pd.DataFrame({'Categorical': ['a'], 'CategoricalNum': [0], 'Numerical': [25]})


@pytest.fixture(scope='session')
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
