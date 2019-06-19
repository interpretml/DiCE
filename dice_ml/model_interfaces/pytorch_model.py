"""Module containing an interface to trained PyTorch model (Future Support)."""

class PyTorchModel:

    def __init__(self, model):
        """Init method

        :param model: trained PyTorch Model.

        """

        self.model = model

    def get_output(self, input):
        # Future Support
        return None

    def get_gradient(self, input):
        # Future Support
        return None
