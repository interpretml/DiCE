"""Module containing an interface to trained Keras Tensorflow model."""

from dice_ml.model_interfaces.base_model import BaseModel
import tensorflow as tf
from tensorflow import keras

class KerasTensorFlowModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='TF1'):
        """Init method

        :param model: trained Keras Sequential Model.
        :param model_path: path to trained model.
        :param backend: "TF1" for TensorFlow 1 and "TF2" for TensorFlow 2.
        """

        super().__init__(model, model_path, backend)

    def load_model(self):
        if self.model_path != '':
            self.model = keras.models.load_model(self.model_path)

    def get_output(self, input_tensor, training=False):
        if self.backend == 'TF2':
            return self.model(input_tensor, training=training)
        else:
            return self.model(input_tensor)

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")
