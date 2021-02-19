import json
import pandas as pd

def json_converter(obj):
    """ Helper function to convert CounterfactualExplanations object to json.
    """
    if isinstance(obj, CounterfactualExplanations):
        return obj.__dict__
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__

def as_counterfactual_explanations(json_dict):
    """ Helper function to convert json string to a CounterfactualExplanations
    object.
    """
    if 'metadata' in json_dict:
        cf_examples_list = []
        for cf_examples_str in json_dict["cf_examples_list"]:
            cf_examples_list.append(pd.read_json(cf_examples_str))
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

    def to_json(self):
        """ Serialize Explanations object to json.
        """
        return json.dumps(self, default=json_converter,
                indent=2)

    @staticmethod
    def from_json(json_str):
        """ Deserialize json string to a CounterfactualExplanations object.
        """
        return json.loads(json_str, object_hook=as_counterfactual_explanations)
