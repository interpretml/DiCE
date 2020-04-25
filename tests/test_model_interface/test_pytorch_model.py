import pytest

import dice_ml
from dice_ml.utils import helpers
import numpy as np

pyt = pytest.importorskip("torch")

@pytest.fixture
def pyt_model_object():
    backend = 'PYT'
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    return m

def test_model_initiation(pyt_model_object):
    assert isinstance(pyt_model_object, dice_ml.model_interfaces.pytorch_model.PyTorchModel)

def test_model_initiation_fullpath():
    pyt = pytest.importorskip("torch")
    backend = {'model': 'pytorch_model.PyTorchModel',
            'explainer': 'dice_pytorch.DicePyTorch'}
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    assert isinstance(m, dice_ml.model_interfaces.pytorch_model.PyTorchModel)

class TestPyTorchModelMethods:
    @pytest.fixture(autouse=True)
    def _get_model_object(self, pyt_model_object):
        self.m = pyt_model_object

    def test_load_model(self):
        self.m.load_model()
        assert self.m.model is not None

    def test_model_output(self):
        self.m.load_model()
        test_instance = pyt.tensor(np.array([[0.5]*29], dtype=np.float32))
        prediction = self.m.get_output(test_instance).detach().numpy()[0][0]
        assert (round(prediction,4)-0.0957)<1e-6
