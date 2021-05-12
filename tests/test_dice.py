import pytest

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException


class TestBaseExplainerLoader:
    def _get_exp(self, backend, method="random", diceml_data=None):
        ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
        m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
        exp = dice_ml.Dice(diceml_data, m, method=method)
        return exp

    def test_tf(self, public_data_object):
        tf = pytest.importorskip("tensorflow")
        backend = 'TF'+tf.__version__[0]
        exp = self._get_exp(backend, diceml_data=public_data_object)
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_tensorflow2.DiceTensorFlow2) or \
            isinstance(exp, dice_ml.explainer_interfaces.dice_tensorflow1.DiceTensorFlow1)

    def test_pyt(self, public_data_object):
        pytest.importorskip("torch")
        backend = 'PYT'
        exp = self._get_exp(backend, diceml_data=public_data_object)
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_pytorch.DicePyTorch)

    @pytest.mark.parametrize('method', ['random'])
    def test_sklearn(self, method, public_data_object):
        pytest.importorskip("sklearn")
        backend = 'sklearn'
        exp = self._get_exp(backend, method=method, diceml_data=public_data_object)
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)
        assert isinstance(exp, dice_ml.explainer_interfaces.dice_random.DiceRandom)

    def test_minimum_query_instances(self, public_data_object, sample_adultincome_query):
        pytest.importorskip('sklearn')
        backend = 'sklearn'
        exp = self._get_exp(backend, diceml_data=public_data_object)
        with pytest.raises(UserConfigValidationException):
            exp.global_feature_importance(sample_adultincome_query)

    def test_unsupported_sampling_strategy(self, public_data_object):
        pytest.importorskip('sklearn')
        backend = 'sklearn'
        with pytest.raises(UserConfigValidationException):
            self._get_exp(backend, method="unsupported", diceml_data=public_data_object)
