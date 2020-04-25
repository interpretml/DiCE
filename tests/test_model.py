import pytest

import dice_ml
from dice_ml.utils import helpers

class TestBaseModelLoader:
    def _get_model(self, backend):
        ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
        m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
        return m

    def test_tf(self):
        tf = pytest.importorskip("tensorflow")
        backend = 'TF'+tf.__version__[0]
        m = self._get_model(backend)
        assert issubclass(type(m), dice_ml.model_interfaces.base_model.BaseModel)

    def test_pyt(self):
        pyt = pytest.importorskip("torch")
        backend = 'PYT'
        m = self._get_model(backend)
        assert issubclass(type(m), dice_ml.model_interfaces.base_model.BaseModel)
