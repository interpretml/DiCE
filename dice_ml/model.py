"""Module pointing to different implementations of Model class

The implementations contain methods to access the output or gradients of ML models trained based on different frameworks such as Tensorflow or PyTorch.
"""

class Model:
    """An interface class to different ML Model implementations."""
    def __init__(self, model=None, model_path='', backend='TF1'):
        """Init method

        :param model: trained ML model.
        :param model_path: path to trained ML model.
        :param backend: tensorflow 1.0/2.0 or pytorch framework.

        """
        if((model is None) & (model_path == '')):
            raise ValueError("should provide either a trained model or the path to a model")
        else:
            self.decide_implementation_type(model, model_path, backend)

    def decide_implementation_type(self, model, model_path, backend):
        """Decides the Model implementation type."""

        self.__class__  = decide(backend)
        self.__init__(model, model_path, backend)

# To add new implementations of Model, add the class in model_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.

def decide(backend):
    """Decides the Model implementation type."""

    if 'TF' in backend: # Tensorflow 1 or 2 backend
        from dice_ml.model_interfaces.keras_tensorflow_model import KerasTensorFlowModel
        return KerasTensorFlowModel

    elif backend == 'PYT': # PyTorch backend
        from dice_ml.model_interfaces.pytorch_model import PyTorchModel
        return PyTorchModel
