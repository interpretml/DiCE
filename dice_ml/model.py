"""Module pointing to different implementations of Model class

The implementations contain methods to access the output or gradients of ML models trained based on different frameworks such as Tensorflow or PyTorch.
"""

import tensorflow as tf


class Model:
    """An interface class to different ML Model implementations."""
    def __init__(self, model=None, model_path=''):
        """Init method

        :param model: trained ML model.
        :param model_path: path to trained ML model.

        """
        if((model is None) & (model_path == '')):
            raise ValueError("should provide either a trained model or the path to a model")
        else:
            self.decide_implementation_type(model, model_path)

    def decide_implementation_type(self, model, model_path):
        """Decides the Model implementation type."""

        self.__class__  = decide(model, model_path)
        self.__init__(model, model_path)

# To add new implementations of Model, add the class in model_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.

def decide(model, model_path):
    """Decides the Model implementation type."""

    if( (isinstance(model, tf.keras.models.Sequential)) | (model_path.endswith('.h5')) ): # pretrained Keras Sequential model with Tensorflow backend
        from dice_ml.model_interfaces.keras_tensorflow_model import KerasTensorFlowModel
        return KerasTensorFlowModel

    elif(isinstance(model, type)): # Future Support: PyTorch model class
        from dice_ml.model_interfaces.pytorch_model import PyTorchModel
        return PyTorchModel
