import numpy as np
import pytest

import dice_ml
from dice_ml.utils import helpers

tf = pytest.importorskip("tensorflow")

@pytest.fixture
def tf_session():
    if tf.__version__[0] == '1':
        sess = tf.InteractiveSession()
        return sess

@pytest.fixture
def tf_model_object():
    backend = 'TF'+tf.__version__[0]
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    return m

def test_model_initiation(tf_model_object):
    assert isinstance(tf_model_object, dice_ml.model_interfaces.keras_tensorflow_model.KerasTensorFlowModel)

def test_model_initiation_fullpath():
    """
    Tests if model is initiated when full path to a model and explainer class is given to backend parameter.
    """
    tf_version = tf.__version__[0]
    backend = {'model': 'keras_tensorflow_model.KerasTensorFlowModel',
            'explainer': 'dice_tensorflow'+tf_version+'.DiceTensorFlow'+tf_version}
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    assert isinstance(m, dice_ml.model_interfaces.keras_tensorflow_model.KerasTensorFlowModel)

class TestKerasModelMethods:
    @pytest.fixture(autouse=True)
    def _get_model_object(self, tf_model_object, tf_session):
        self.m = tf_model_object
        self.sess = tf_session

    def test_load_model(self):
        self.m.load_model()
        assert self.m.model is not None

    @pytest.mark.parametrize("input_instance, prediction",[(np.array([[0.5]*29], dtype=np.float32), 0.747)])
    def test_model_output(self, input_instance, prediction):
        self.m.load_model()
        if tf.__version__[0] == '1':
            input_instance_tf = tf.Variable(input_instance, dtype=tf.float32)
            output_instance = self.m.get_output(input_instance_tf)
            prediction = self.sess.run(output_instance, feed_dict={input_instance_tf:input_instance})[0][0]
        else:
            prediction = self.m.get_output(input_instance).numpy()[0][0]
        pytest.approx(prediction, abs=1e-3) == prediction
