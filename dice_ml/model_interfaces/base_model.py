"""Module containing a template class as an interface to ML model.
   Subclasses implement model interfaces for different ML frameworks such as TensorFlow, PyTorch OR Sklearn.
   All model interface methods are in dice_ml.model_interfaces"""

import pickle

import numpy as np

from dice_ml.constants import ModelTypes
from dice_ml.utils.exception import SystemException
from dice_ml.utils.helpers import DataTransfomer


class BaseModel:

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch,
                        or for implementations other than standard DiCE
                        (https://arxiv.org/pdf/1905.07697.pdf),
                        provide both the module and class names as module_name.class_name.
                        For instance, if there is a model interface class "SklearnModel"
                        in module "sklearn_model.py" inside the subpackage dice_ml.model_interfaces,
                        then backend parameter should be "sklearn_model.SklearnModel".
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.

        """
        self.model = model
        self.model_path = model_path
        self.backend = backend
        # calls FunctionTransformer of scikit-learn internally
        # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
        self.transformer = DataTransfomer(func, kw_args)

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance)
        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.model.predict_proba(input_instance)
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]
