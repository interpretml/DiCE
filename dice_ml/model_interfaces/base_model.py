"""Module containing a template class as an interface to ML model.
   Subclasses implement model interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All model interface methods are in dice_ml.model_interfaces"""

import pickle

class BaseModel:

    def __init__(self, model=None, model_path='', backend=''):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch, or for implementations other than standard DiCE (https://arxiv.org/pdf/1905.07697.pdf), provide both the module and class names as module_name.class_name. For instance, if there is a model interface class "SklearnModel" in module "sklearn_model.py" inside the subpackage dice_ml.model_interfaces, then backend parameter should be "sklearn_model.SklearnModel".
        """

        self.model = model
        self.model_path = model_path
        self.backend = backend

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def get_output(self, input_instance):
        """returns prediction probabilities"""
        return self.model.predict_proba(input_instance)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self):
        return 0
