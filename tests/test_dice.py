import pytest

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException


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
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)

    def test_pyt(self):
        pyt = pytest.importorskip("torch")
        backend = 'PYT'
        exp = self._get_exp(backend)
        assert issubclass(type(exp), dice_ml.explainer_interfaces.explainer_base.ExplainerBase)

    def test_minimum_query_instances(self):
        sklearn_model = pytest.importorskip('sklearn')
        backend = 'sklearn'
        exp = self._get_exp(backend)
        with pytest.raises(UserConfigValidationException):
            query_instances = helpers.load_adult_income_dataset().drop("income", axis=1)[0:1]
            exp.global_feature_importance(query_instances)
