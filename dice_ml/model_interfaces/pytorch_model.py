"""Module containing an interface to trained PyTorch model."""

class PyTorchModel:

    def __init__(self, model):
        """Init method

        Args:
            model: trained PyTorch Model.
        """
        
        self.model = model

    def get_output(self, input):
        # TODO:
        return None

    def get_gradient(self, input):
        # TODO:
        return None
