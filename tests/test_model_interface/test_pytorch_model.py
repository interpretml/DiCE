import numpy as np
import pytest

import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.helpers import DataTransfomer

pyt = pytest.importorskip("torch")

@pytest.fixture
def pyt_model_object():
    backend = 'PYT'
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend, func='ohe-min-max')
    return m

def test_model_initiation(pyt_model_object):
    assert isinstance(pyt_model_object, dice_ml.model_interfaces.pytorch_model.PyTorchModel)

def test_model_initiation_fullpath():
    """
    Tests if model is initiated when full path to a model and explainer class is given to backend parameter.
    """
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

    # @pytest.mark.parametrize("input_instance, prediction",[(np.array([[0.5]*29], dtype=np.float32), 0.0957)])
    # def test_model_output(self, input_instance, prediction):
    #     self.m.load_model()
    #     input_instance_pyt = pyt.tensor(input_instance)
    #     prediction = self.m.get_output(input_instance_pyt).detach().numpy()[0][0]
    #     pytest.approx(prediction, abs=1e-3) == prediction
    @pytest.mark.parametrize("prediction",[0.0957])
    def test_model_output(self, sample_adultincome_query, public_data_object, prediction):
        # initializing data transormation required for ML model
        public_data_object.create_ohe_params()
        self.m.load_model()
        self.m.transformer = DataTransfomer(func='ohe-min-max', kw_args=None)
        self.m.transformer.feed_data_params(public_data_object)
        self.m.transformer.initialize_transform_func()
        output_instance = self.m.get_output(sample_adultincome_query, transform_data=True)
        predictval = output_instance.detach().numpy()[0][0]
        pytest.approx(predictval, abs=1e-3) == prediction
