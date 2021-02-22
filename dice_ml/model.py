"""Module pointing to different implementations of Model class

The implementations contain methods to access the output or gradients of ML models trained based on different frameworks such as Tensorflow or PyTorch.
"""

class Model:
    """An interface class to different ML Model implementations."""
    def __init__(self, model=None, model_path='', backend='TF1', model_type='classifier', func=None, kw_args=None):
        """Init method

        :param model: trained ML model.
        :param model_path: path to trained ML model.
        :param backend: "TF1" ("TF2") for TensorFLow 1.0 (2.0), "PYT" for PyTorch implementations, "sklearn" for Scikit-Learn implementations of standard DiCE (https://arxiv.org/pdf/1905.07697.pdf). For all other frameworks and implementations, provide a dictionary with "model" and "explainer" as keys, and include module and class names as values in the form module_name.class_name. For instance, if there is a model interface class "XGBoostModel" in module "xgboost_model.py" inside the subpackage dice_ml.model_interfaces, and dice interface class "DiceXGBoost" in module "dice_xgboost" inside dice_ml.explainer_interfaces, then backend parameter should be {"model": "xgboost_model.XGBoostModel", "explainer": dice_xgboost.DiceXGBoost}.
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the dictionary of kw_args, by default.
        """
        self.model_type = model_type
        if((model is None) & (model_path == '')):
            raise ValueError("should provide either a trained model or the path to a model")
        else:
            self.decide_implementation_type(model, model_path, backend, func, kw_args)

    def decide_implementation_type(self, model, model_path, backend, func, kw_args):
        """Decides the Model implementation type."""

        self.__class__  = decide(backend)
        self.__init__(model, model_path, backend, func, kw_args)

# To add new implementations of Model, add the class in model_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.

def decide(backend):
    """Decides the Model implementation type."""

    if backend == 'sklearn': # random sampling of CFs
        from dice_ml.model_interfaces.base_model import BaseModel
        return BaseModel

    elif 'TF' in backend: # Tensorflow 1 or 2 backend
        from dice_ml.model_interfaces.keras_tensorflow_model import KerasTensorFlowModel
        return KerasTensorFlowModel

    elif backend == 'PYT': # PyTorch backend
        from dice_ml.model_interfaces.pytorch_model import PyTorchModel
        return PyTorchModel

    else: # all other implementations and frameworks
        backend_model = backend['model']
        module_name, class_name = backend_model.split('.')
        module = __import__("dice_ml.model_interfaces." + module_name, fromlist=[class_name])
        return getattr(module, class_name)
