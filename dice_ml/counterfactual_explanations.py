import json
import pandas as pd

import dice_ml.utils.serialize
import dice_ml.diverse_counterfactuals as counterfactual_examples

def as_counterfactual_explanations(json_dict):
    """ Helper function to convert json string to a CounterfactualExplanations
    object.
    """
    if 'metadata' in json_dict:
        cf_examples_list = []
        for cf_examples_str in json_dict["cf_examples_list"]:
            cf_examples_dict = json.loads(cf_examples_str)
            test_instance_df = pd.read_json(cf_examples_dict["test_instance_df"])
            cfs_df = pd.read_json(cf_examples_dict["final_cfs_df"])
            cf_examples_list.append(
                    counterfactual_examples.CounterfactualExamples(data_interface=None,
                                          test_instance_df=test_instance_df,
                                          final_cfs_df=cfs_df,
                                          final_cfs_df_sparse=cfs_df,
                                          posthoc_sparsity_param=None,
                                          desired_class=cf_examples_dict["desired_class"],
                                          desired_range=cf_examples_dict["desired_range"],
                                          model_type=cf_examples_dict["model_type"])
                    )
        return CounterfactualExplanations(cf_examples_list,
                local_importance=json_dict["local_importance"],
                summary_importance=json_dict["summary_importance"])

    else:
        return json_dict

class CounterfactualExplanations:
    """A class to store counterfactual examples for one or more inputs
    and feature importance scores.

    :param cf_examples_list: A list of CounterfactualExamples instances
    :param local_importance: List of estimated local importance scores. The
    size of the list is the number of input instances, each containing feature
    importance scores for that input.
    :param summary_importance: Estimated global feature importance scores
    based on the input set of CounterfactualExamples instances

    """
    def __init__(self, cf_examples_list,
                 local_importance=None,
                 summary_importance=None):
        self.cf_examples_list = cf_examples_list
        self.local_importance = local_importance
        self.summary_importance = summary_importance
        self.metadata = {'version': '1'}

    def visualize_as_dataframe(self, display_sparse_df=True,
                               show_only_changes=False):
        for cf_examples in self.cf_examples_list:
            cf_examples.visualize_as_dataframe(
                    display_sparse_df=display_sparse_df,
                    show_only_changes=show_only_changes)

    def visualize_as_list(self, display_sparse_df=True,
                          show_only_changes=False):
        for cf_examples in self.cf_examples_list:
            cf_examples.visualize_as_list(
                    display_sparse_df=display_sparse_df,
                    show_only_changes=show_only_changes)

    def to_json(self):
        """ Serialize Explanations object to json.
        """
        return json.dumps(self, default=dice_ml.utils.serialize.json_converter,
                          indent=2)

    @staticmethod
    def from_json(json_str):
        """ Deserialize json string to a CounterfactualExplanations object.
        """
        return json.loads(json_str, object_hook=as_counterfactual_explanations)
