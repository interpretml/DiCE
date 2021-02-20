"""Module pointing to different implementations of DiCE based on different frameworks such as Tensorflow or PyTorch or sklearn, and different methods such as RandomSampling, DiCEKD or DiCEGenetic"""

class Dice:
    """An interface class to different DiCE implementations."""

    def __init__(self, data_interface, model_interface, method="genetic",  **kwargs):
        """Init method

        :param data_interface: an interface to access data related params.
        :param model_interface: an interface to access the output or gradients of a trained ML model.
        :param method: Name of the method to use for generating counterfactuals

        """

        self.decide_implementation_type(data_interface, model_interface, method, **kwargs)

    def decide_implementation_type(self, data_interface, model_interface, method, **kwargs):
        """Decides DiCE implementation type."""

        self.__class__  = decide(model_interface, method)
        self.__init__(data_interface, model_interface, **kwargs)

# To add new implementations of DiCE, add the class in explainer_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.

def decide(model_interface, method):
    """Decides DiCE implementation type."""

    if model_interface.backend == 'sklearn': # random sampling of CFs
        if method == "random":
            from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
            return ExplainerBase
        elif method == "genetic":
            from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
            return DiceGenetic
        elif method == "kdtree":
            from dice_ml.explainer_interfaces.dice_KD import DiceKD
            return DiceKD

    elif model_interface.backend == 'TF1': # pretrained Keras Sequential model with Tensorflow 1.x backend
        from dice_ml.explainer_interfaces.dice_tensorflow1 import DiceTensorFlow1
        return DiceTensorFlow1

    elif model_interface.backend == 'TF2': # pretrained Keras Sequential model with Tensorflow 2.x backend
        from dice_ml.explainer_interfaces.dice_tensorflow2 import DiceTensorFlow2
        return DiceTensorFlow2

    elif model_interface.backend == 'PYT': # PyTorch backend
        from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
        return DicePyTorch

    else: # all other backends
        backend_dice = model_interface.backend['explainer']
        module_name, class_name = backend_dice.split('.')
        module = __import__("dice_ml.explainer_interfaces." + module_name, fromlist=[class_name])
        return getattr(module, class_name)
