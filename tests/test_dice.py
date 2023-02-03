import pytest
from raiutils.exceptions import UserConfigValidationException

import dice_ml
from dice_ml.utils import helpers


class TestBaseExplainerLoader:
    def _get_exp(self, backend, method="random", is_public_data_interface=True):
        if is_public_data_interface:
            dataset = helpers.load_adult_income_dataset()
            d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
        else:
            d = dice_ml.Data(features={
                'age': [17, 90],
                'workclass': ['Government', 'Other/Unknown', 'Private', 'Self-Employed'],
                'education': ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',
                              'Prof-school', 'School', 'Some-college'],
                'marital_status': ['Divorced', 'Married', 'Separated', 'Single', 'Widowed'],
                'occupation': ['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar'],
                'race': ['Other', 'White'],
                'gender': ['Female', 'Male'],
                'hours_per_week': [1, 99]},
                            outcome_name='income')
        ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
        m = dice_ml.Model(model_path=ML_modelpath, backend=backend, func="ohe-min-max")
        exp = dice_ml.Dice(d, m, method=method)
        return exp

    def test_tf(self):
        tf = pytest.importorskip("tensorflow")
        backend = 'TF'+tf.__version__[0]
        exp = self._get_exp(backend, method="gradient")
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_tensorflow2.DiceTensorFlow2) or \
            isinstance(exp, dice_ml.explainer_interfaces.dice_tensorflow1.DiceTensorFlow1)

    def test_pyt(self):
        pytest.importorskip("torch")
        backend = 'PYT'
        exp = self._get_exp(backend, method="gradient")
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_pytorch.DicePyTorch)

    @pytest.mark.parametrize('method', ['random'])
    def test_sklearn(self, method):
        pytest.importorskip("sklearn")
        backend = 'sklearn'
        exp = self._get_exp(backend, method=method)
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_random.DiceRandom)

    def test_minimum_query_instances(self):
        pytest.importorskip('sklearn')
        backend = 'sklearn'
        exp = self._get_exp(backend)
        query_instances = helpers.load_adult_income_dataset().drop("income", axis=1)[0:1]
        with pytest.raises(UserConfigValidationException):
            exp.global_feature_importance(query_instances)

    def test_unsupported_sampling_strategy(self):
        pytest.importorskip('sklearn')
        backend = 'sklearn'
        with pytest.raises(UserConfigValidationException):
            self._get_exp(backend, method="unsupported")

    def test_private_data_interface_dice_kdtree(self):
        pytest.importorskip("sklearn")
        backend = 'sklearn'
        with pytest.raises(UserConfigValidationException) as ucve:
            self._get_exp(backend, method='kdtree', is_public_data_interface=False)

        assert 'Private data interface is not supported with kdtree explainer' + \
               ' since kdtree explainer needs access to entire training data' in str(ucve)
