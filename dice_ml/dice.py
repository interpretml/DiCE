"""Module pointing to different implementations of DiCE based on different frameworks such as Tensorflow or PyTorch."""

import tensorflow as tf

class Dice:
    """An interface class to different DiCE implementations."""

    def __init__(self, data_interface, model_interface):
        """Init method

        Args:
            data_interface: an interface to access data related params.
            model_interface: an interface to access the output or gradients of a trained ML model.
        """

        self.decide_implementation_type(data_interface, model_interface)

    def decide_implementation_type(self, data_interface, model_interface):
        """Decides DiCE implementation type."""

        self.__class__  = decide(data_interface, model_interface)
        self.__init__(data_interface, model_interface)

# To add new implementations of DiCE, add the class in dice_interfaces subpackage and import-and-return the class in an elif loop as shown in the below method.
def decide(data_interface, model_interface):
    """Decides DiCE implementation type."""

    if( (isinstance(model_interface.model, tf.keras.models.Sequential)) | (model_interface.model_path.endswith('.h5')) ): # pretrained Keras Sequential model with Tensorflow backend
        from dice_ml.dice_interfaces.dice_tensorflow import DiceTensorFlow
        return DiceTensorFlow

    else: #elif(isinstance(model_interface.model, 'PyTorch')): # TODO: support for PyTorch 
        from dice_ml.dice_interfaces.dice_pytorch import DicePyTorch
        return DicePyTorch
