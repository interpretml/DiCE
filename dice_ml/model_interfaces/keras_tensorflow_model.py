"""Module containing an interface to trained Keras Tensorflow model."""

import tensorflow as tf
from tensorflow import keras

class KerasTensorFlowModel:

    def __init__(self, model=None, model_path=''):
        """Init method

        Args:
            model: trained Keras Sequential Model.
            model_path: path to trained model
        """

        self.model = model
        self.model_path = model_path

    def load_model(self):
        if self.model_path != '':
            self.model = keras.models.load_model(self.model_path)

    def get_output(self, input_tensor):
        return self.model(input_tensor)

    def get_gradient(self, input):
        # Future Support
        return None
