import pytest

import dice_ml
from dice_ml.utils import helpers

class TestBaseExplainerLoader:
    def _get_exp(self, backend):
        dataset = helpers.load_adult_income_dataset()
        d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
        ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
        m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
        exp = dice_ml.Dice(d, m)
        return exp

    def test_tf(self):
        tf = pytest.importorskip("tensorflow")
        backend = 'TF'+tf.__version__[0]
        exp = self._get_exp(backend)
        assert issubclass(type(exp), dice_ml.dice_interfaces.dice_base.DiceBase)

    def test_pyt(self):
        pyt = pytest.importorskip("torch")
        backend = 'PYT'
        exp = self._get_exp(backend)
        assert issubclass(type(exp), dice_ml.dice_interfaces.dice_base.DiceBase)
