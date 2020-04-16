"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice_ml.dice_interfaces"""

class DiceBase:

    def __init__(self, data_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        """
        # get data-related parameters - minx and max for normalized continuous features
        self.data_interface = data_interface
        self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()

        # min and max for continuous features in original scale
        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        org_minx, org_maxx = self.data_interface.get_minx_maxx(normalized=False)
        self.cont_minx = list(org_minx[0][self.encoded_continuous_feature_indexes])
        self.cont_maxx = list(org_maxx[0][self.encoded_continuous_feature_indexes])

        # decimal precisions for continuous features
        self.cont_precisions = [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def generate_counterfactuals(self):
        raise NotImplementedError
