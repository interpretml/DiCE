"""Module containing an interface to trained PyTorch model."""

import numpy as np
import torch

from dice_ml.constants import ModelTypes
from dice_ml.model_interfaces.base_model import BaseModel


class PyTorchModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='PYT', func=None, kw_args=None):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        :param backend: "PYT" for PyTorch framework.
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.
        """

        super().__init__(model, model_path, backend, func, kw_args)

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_instance, model_score=True,
                   transform_data=False, out_tensor=False):
        """returns prediction probabilities

        :param input_tensor: test input.
        :param transform_data: boolean to indicate if data transformation is required.
        """
        input_tensor = input_instance
        if transform_data:
            input_tensor = torch.tensor(self.transformer.transform(input_instance).to_numpy()).float()
        if not torch.is_tensor(input_instance):
            input_tensor = torch.tensor(self.transformer.transform(input_instance).to_numpy()).float()
        out = self.model(input_tensor).float()
        if not out_tensor:
            out = out.data.numpy()
        if model_score is False and self.model_type == ModelTypes.Classifier:
            out = np.round(out)  # TODO need to generalize for n-class classifier
        return out

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input_instance):
        # Future Support
        raise NotImplementedError("Future Support")

    def get_num_output_nodes(self, inp_size):
        temp_input = torch.rand(1, inp_size).float()
        return self.get_output(temp_input).data
