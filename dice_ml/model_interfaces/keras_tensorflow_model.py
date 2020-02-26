"""Module containing an interface to trained Keras Tensorflow model."""

import tensorflow as tf
from tensorflow import keras

class KerasTensorFlowModel:

    def __init__(self, model=None, model_path='', backend='TF1'):
        """Init method

        :param model: trained Keras Sequential Model.
        :param model_path: path to trained model.
        :param backend: tensorflow 1.0/2.0 or pytorch framework.
        """

        self.model = model
        self.model_path = model_path
        self.backend = backend

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
        return None
