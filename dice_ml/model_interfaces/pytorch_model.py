"""Module containing an interface to trained PyTorch model (Future Support)."""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class PyTorchModel(nn.Module):

    def __init__(self, inp_shape):
        """Init method

        :param model: trained PyTorch Model.

        """

        super(PyTorchModel, self).__init__()
        self.inp_shape= inp_shape
        self.hidden_dim= 20
        self.num_classes=2
        self.ann_model= nn.Sequential(
                        nn.Linear( self.inp_shape, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear( self.hidden_dim, self.num_classes),
                        nn.Softmax(),
                )
    
    def forward(self, x):
        return self.ann_model(x)
       
    def get_output(self, input):
        # Future Support
        return None

    def get_gradient(self, input):
        # Future Support
        return None
