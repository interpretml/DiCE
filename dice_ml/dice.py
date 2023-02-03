"""Module pointing to different implementations of DiCE based on different
   frameworks such as Tensorflow or PyTorch or sklearn, and different methods
   such as RandomSampling, DiCEKD or DiCEGenetic"""

from raiutils.exceptions import UserConfigValidationException

from dice_ml.constants import BackEndTypes, SamplingStrategy
from dice_ml.data_interfaces.private_data_interface import PrivateData
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class Dice(ExplainerBase):
    """An interface class to different DiCE implementations."""

    def __init__(self, data_interface, model_interface, method="random",  **kwargs):
        """Init method

        :param data_interface: an interface to access data related params.
        :param model_interface: an interface to access the output or gradients of a trained ML model.
        :param method: Name of the method to use for generating counterfactuals
        """
        self.decide_implementation_type(data_interface, model_interface, method, **kwargs)

    def decide_implementation_type(self, data_interface, model_interface, method, **kwargs):
        """Decides DiCE implementation type."""
        if model_interface.backend == BackEndTypes.Sklearn:
            if method == SamplingStrategy.KdTree and isinstance(data_interface, PrivateData):
                raise UserConfigValidationException(
                    'Private data interface is not supported with kdtree explainer'
                    ' since kdtree explainer needs access to entire training data')
        self.__class__ = decide(model_interface, method)
        self.__init__(data_interface, model_interface, **kwargs)

    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  permitted_range=None, features_to_vary="all",
                                  stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="linear", verbose=False, **kwargs):
        raise NotImplementedError("This method should be implemented by the concrete classes "
                                  "that inherit from ExplainerBase")


def decide(model_interface, method):
    """Decides DiCE implementation type.

    To add new implementations of DiCE, add the class in explainer_interfaces
    subpackage and import-and-return the class in an elif loop as shown in
    the below method.
    """
    if method == SamplingStrategy.Random:
        # random sampling of CFs
        from dice_ml.explainer_interfaces.dice_random import DiceRandom
        return DiceRandom
    elif method == SamplingStrategy.Genetic:
        from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic
        return DiceGenetic
    elif method == SamplingStrategy.KdTree:
        from dice_ml.explainer_interfaces.dice_KD import DiceKD
        return DiceKD
    elif method == SamplingStrategy.Gradient:
        if model_interface.backend == BackEndTypes.Tensorflow1:
            # pretrained Keras Sequential model with Tensorflow 1.x backend
            from dice_ml.explainer_interfaces.dice_tensorflow1 import \
                DiceTensorFlow1
            return DiceTensorFlow1

        elif model_interface.backend == BackEndTypes.Tensorflow2:
            # pretrained Keras Sequential model with Tensorflow 2.x backend
            from dice_ml.explainer_interfaces.dice_tensorflow2 import \
                DiceTensorFlow2
            return DiceTensorFlow2

        elif model_interface.backend == BackEndTypes.Pytorch:
            # PyTorch backend
            from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
            return DicePyTorch
        else:
            raise UserConfigValidationException(
                    "{0} is only supported for differentiable neural network models. "
                    "Please choose one of {1}, {2} or {3}".format(
                        method, SamplingStrategy.Random,
                        SamplingStrategy.Genetic,
                        SamplingStrategy.KdTree
                    ))
    elif method is None:
        # all other backends
        backend_dice = model_interface.backend['explainer']
        module_name, class_name = backend_dice.split('.')
        module = __import__("dice_ml.explainer_interfaces." + module_name, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise UserConfigValidationException("Unsupported sample strategy {0} provided. "
                                            "Please choose one of {1}, {2} or {3}".format(
                                                method, SamplingStrategy.Random,
                                                SamplingStrategy.Genetic,
                                                SamplingStrategy.KdTree
                                            ))
