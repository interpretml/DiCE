import json
import pandas as pd

import dice_ml.diverse_counterfactuals as exp
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.utils.exception import UserConfigValidationException


_CurrentVersion = '2.0'
_AllVersions = [_CurrentVersion, '1.0']


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

        if version == '1.0':
            cf_examples_list = []
            for cf_examples_str in json_dict["cf_examples_list"]:
                cf_examples_list.append(CounterfactualExamples.from_json(cf_examples_str))

            return CounterfactualExplanations(
                    cf_examples_list=cf_examples_list,
                    local_importance=json_dict["local_importance"],
                    summary_importance=json_dict["summary_importance"])
        elif version == '2.0':
            cf_examples_list = []
            for index in range(0, len(json_dict['cfs_list'])):
                cf_examples_list.append(
                    CounterfactualExamples.from_json_v2(
                        json_dict['cfs_list'][index],
                        json_dict['test_data'][index],
                        json_dict['metadata']
                    )
                )
            local_importance_list = None
            if json_dict['local_importance'] is not None:
                local_importance_list = []
                for local_importance_instance in json_dict['local_importance']:
                    local_importance_dict = {}
                    feature_names = json_dict['metadata']['feature_names']
                    for index in range(0, len(local_importance_instance)):
                        local_importance_dict[feature_names[index]] = local_importance_instance[index]
                    local_importance_list.append(local_importance_dict)

            summary_importance_dict = None
            if json_dict['summary_importance'] is not None:
                summary_importance_dict = {}
                feature_names = json_dict['metadata']['feature_names']
                for index in range(0, len(json_dict['summary_importance'])):
                    summary_importance_dict[feature_names[index]] = json_dict['summary_importance'][index]

            return CounterfactualExplanations(
                    cf_examples_list=cf_examples_list,
                    local_importance=local_importance_list,
                    summary_importance=summary_importance_dict)
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
        if len(self.cf_examples_list) > 0:
            serialization_version = self.cf_examples_list[0]._get_serialiation_version()
            if serialization_version == '1.0':
                return json.dumps(self, default=json_converter,
                                  indent=2)
            elif serialization_version == '2.0':
                combined_test_instance_list = []
                combined_final_dfs_list = []
                metadata = None
                for cf_examples in self.cf_examples_list:
                    cf_examples_str = cf_examples.to_json()
                    serialized_cf_examples = json.loads(cf_examples_str)
                    combined_test_instance_list.append(serialized_cf_examples['test_instance_list'])
                    combined_final_dfs_list.append(serialized_cf_examples['final_dfs_list'])
                    metadata = serialized_cf_examples['metadata']

                if metadata is None:
                    raise Exception("No counterfactual examples found")

                local_importance_matrix = None
                if self.local_importance is not None:
                    local_importance_matrix = []
                    for local_importance_dict in self.local_importance:
                        local_importance_list = []
                        for feature_name in metadata['feature_names']:
                            local_importance_list.append(local_importance_dict.get(feature_name))
                        local_importance_matrix.append(local_importance_list)

                summary_importance_list = None
                if self.summary_importance is not None:
                    summary_importance_list = []
                    for feature_name in metadata['feature_names']:
                        summary_importance_list.append(self.summary_importance.get(feature_name))

                entire_dict = {
                    'test_data': combined_test_instance_list,
                    'cfs_list': combined_final_dfs_list,
                    'local_importance': local_importance_matrix,
                    'summary_importance': summary_importance_list,
                    'metadata': metadata
                }
                return json.dumps(entire_dict)
            else:
                raise Exception("Unsupported serialization version {}".format(
                    serialization_version))

    @staticmethod
    def from_json(json_str):
        """ Deserialize json string to a CounterfactualExplanations object.
        """
        return json.loads(json_str, object_hook=as_counterfactual_explanations)
