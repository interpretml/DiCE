"""Module containing an interface to trained PyTorch model."""

import torch

class PyTorchModel:

    def __init__(self, model=None, model_path='', backend='PYT'):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        :param backend: tensorflow 1.0/2.0 or pytorch framework.
        """

        self.model = model
        self.model_path = model_path
        self.backend = backend
        self.model.load_state_dict(model_path)
        self.model.eval()

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_tensor):
        return self.model(input_tensor).float()

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input):
        # Future Support
        return None
    
    def forward(self, x):
        return self.model(x)
