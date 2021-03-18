import json
import pandas as pd

import dice_ml.diverse_counterfactuals as exp
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.utils.exception import UserConfigValidationException


_CurrentVersion = '1.0'
_AllVersions = [_CurrentVersion]


def _check_supported_json_output_versions(version):
    return version in _AllVersions

def json_converter(obj):
    """ Helper function to convert CounterfactualExplanations object to json.
    """
    if isinstance(obj, CounterfactualExplanations):
        rdict = obj.__dict__
        return rdict
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__


def as_counterfactual_explanations(json_dict):
    """ Helper function to convert json string to a CounterfactualExplanations
    object.
    """
    if 'metadata' in json_dict:
        version = json_dict['metadata'].get('version')
        if version is None:
            raise UserConfigValidationException("No version field in the json input")
        elif not _check_supported_json_output_versions(version):
            raise UserConfigValidationException("Incompatible version {} found in json input".format(version))

        cf_examples_list = []
        for cf_examples_str in json_dict["cf_examples_list"]:
            cf_examples_list.append(CounterfactualExamples.from_json(cf_examples_str))

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
        self._cf_examples_list = cf_examples_list
        self._local_importance = local_importance
        self._summary_importance = summary_importance
        self._metadata = {'version': _CurrentVersion}

    def __eq__(self, other_cf):
        if (isinstance(other_cf, CounterfactualExplanations)):
            return self.cf_examples_list == other_cf.cf_examples_list and \
                    self.local_importance == other_cf.local_importance and \
                    self.summary_importance == other_cf.summary_importance and \
                    self.metadata == other_cf.metadata
        return False

    @property
    def __dict__(self):
        return {'cf_examples_list': self.cf_examples_list,
                'local_importance': self.local_importance,
                'summary_importance': self.summary_importance,
                'metadata': self.metadata}

    @property
    def cf_examples_list(self):
        return self._cf_examples_list

    @property
    def local_importance(self):
        if isinstance(self._local_importance, list):
            sorted_local_importance = []
            for local_importance_instance in self._local_importance:
                local_importance_instance = \
                    dict(sorted(local_importance_instance.items(),
                                key=lambda x: x[1], reverse=True))
                sorted_local_importance.append(local_importance_instance)
            self._local_importance = sorted_local_importance
        return self._local_importance

    @property
    def summary_importance(self):
        if isinstance(self._summary_importance, dict):
            self._summary_importance = \
                dict(sorted(self._summary_importance.items(),
                            key=lambda x: x[1], reverse=True))
        return self._summary_importance

    @property
    def metadata(self):
        return self._metadata

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
        return json.dumps(self, default=json_converter,
                          indent=2)

    @staticmethod
    def from_json(json_str):
        """ Deserialize json string to a CounterfactualExplanations object.
        """
        return json.loads(json_str, object_hook=as_counterfactual_explanations)
