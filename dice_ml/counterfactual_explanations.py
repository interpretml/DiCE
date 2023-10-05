import json
import os

import jsonschema
from raiutils.exceptions import UserConfigValidationException

from dice_ml.constants import _SchemaVersions
from dice_ml.diverse_counterfactuals import (CounterfactualExamples,
                                             _DiverseCFV2SchemaConstants)


class _CommonSchemaConstants:
    LOCAL_IMPORTANCE = 'local_importance'
    SUMMARY_IMPORTANCE = 'summary_importance'
    METADATA = 'metadata'


class _CounterfactualExpV1SchemaConstants:
    CF_EXAMPLES_LIST = 'cf_examples_list'
    LOCAL_IMPORTANCE = _CommonSchemaConstants.LOCAL_IMPORTANCE
    SUMMARY_IMPORTANCE = _CommonSchemaConstants.SUMMARY_IMPORTANCE
    METADATA = _CommonSchemaConstants.METADATA


class _CounterfactualExpV2SchemaConstants:
    TEST_DATA = 'test_data'
    CFS_LIST = 'cfs_list'
    LOCAL_IMPORTANCE = _CommonSchemaConstants.LOCAL_IMPORTANCE
    SUMMARY_IMPORTANCE = _CommonSchemaConstants.SUMMARY_IMPORTANCE
    METADATA = _CommonSchemaConstants.METADATA
    MODEL_TYPE = 'model_type'
    DATA_INTERFACE = 'data_interface'
    FEATURE_NAMES = 'feature_names'
    DESIRED_CLASS = 'desired_class'
    DESIRED_RANGE = 'desired_range'
    FEATURE_NAMES_INCLUDING_TARGET = 'feature_names_including_target'


def _check_supported_json_output_versions(version):
    return version in _SchemaVersions.ALL_VERSIONS


class CounterfactualExplanations:
    """A class to store counterfactual examples for one or more inputs
    and feature importance scores.

    :param cf_examples_list: A list of CounterfactualExamples instances
    :param local_importance: List of estimated local importance scores. The
        size of the list is the number of input instances, each containing
        feature importance scores for that input.
    :param summary_importance: Estimated global feature importance scores
        based on the input set of CounterfactualExamples instances

    """
    def __init__(self, cf_examples_list,
                 local_importance=None,
                 summary_importance=None,
                 version=None):
        self._cf_examples_list = cf_examples_list
        self._local_importance = local_importance
        self._summary_importance = summary_importance
        self._metadata = {'version': version if version is not None else _SchemaVersions.CURRENT_VERSION}

    def __eq__(self, other_cf):
        if isinstance(other_cf, CounterfactualExplanations):
            return self.cf_examples_list == other_cf.cf_examples_list and \
                    self.local_importance == other_cf.local_importance and \
                    self.summary_importance == other_cf.summary_importance and \
                    self.metadata == other_cf.metadata
        return False

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

    @staticmethod
    def _check_cf_exp_output_against_json_schema(
            cf_dict, version):
        """
        Validate the dictionary version of the counterfactual explanations.

        :param cf_dict: Serialized version of the counterfactual explanations.
        :type cf_dict: Dict

        """
        schema_file_name = 'counterfactual_explanations_v{0}.json'.format(version)
        schema_path = os.path.join(os.path.dirname(__file__),
                                   'schema', schema_file_name)
        with open(schema_path, 'r') as schema_file:
            schema_json = json.load(schema_file)

        jsonschema.validate(cf_dict, schema_json)

    def to_json(self):
        """ Serialize Explanations object to json.
        """
        serialization_version = self.metadata['version']
        if serialization_version == _SchemaVersions.V1:
            cf_examples_str_list = []
            for cf_examples in self.cf_examples_list:
                cf_examples_str = cf_examples.to_json(
                    serialization_version=serialization_version)
                cf_examples_str_list.append(cf_examples_str)
            entire_dict = {
                _CounterfactualExpV1SchemaConstants.CF_EXAMPLES_LIST: cf_examples_str_list,
                _CounterfactualExpV1SchemaConstants.LOCAL_IMPORTANCE: self.local_importance,
                _CounterfactualExpV1SchemaConstants.SUMMARY_IMPORTANCE: self.summary_importance,
                _CounterfactualExpV1SchemaConstants.METADATA: self.metadata
            }
            CounterfactualExplanations._check_cf_exp_output_against_json_schema(
                entire_dict, version=serialization_version)
            return json.dumps(entire_dict)
        elif serialization_version == _SchemaVersions.V2:
            combined_test_instance_list = []
            combined_final_cfs_list = []
            data_interface = None
            feature_names = None
            feature_names_including_target = None
            model_type = None
            desired_class = None
            desired_range = None
            for cf_examples in self.cf_examples_list:
                cf_examples_str = cf_examples.to_json(
                    serialization_version=serialization_version)
                # We need to load the json again since we need to decompose the
                # counterfactual example into different schema fields
                serialized_cf_examples = json.loads(cf_examples_str)
                combined_test_instance_list.append(serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.TEST_INSTANCE_LIST])
                combined_final_cfs_list.append(serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.FINAL_CFS_LIST])
                data_interface = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.DATA_INTERFACE]
                feature_names = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.FEATURE_NAMES]
                feature_names_including_target = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET]
                model_type = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.MODEL_TYPE]
                desired_class = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.DESIRED_CLASS]
                desired_range = serialized_cf_examples[
                    _DiverseCFV2SchemaConstants.DESIRED_RANGE]

            local_importance_matrix = None
            if self.local_importance is not None:
                local_importance_matrix = []
                for local_importance_dict in self.local_importance:
                    local_importance_list = []
                    for feature_name in feature_names:
                        local_importance_list.append(local_importance_dict.get(feature_name))
                    local_importance_matrix.append(local_importance_list)

            summary_importance_list = None
            if self.summary_importance is not None:
                summary_importance_list = []
                for feature_name in feature_names:
                    summary_importance_list.append(self.summary_importance.get(feature_name))

            entire_dict = {
                _CounterfactualExpV2SchemaConstants.TEST_DATA: combined_test_instance_list,
                _CounterfactualExpV2SchemaConstants.CFS_LIST: combined_final_cfs_list,
                _CounterfactualExpV2SchemaConstants.LOCAL_IMPORTANCE: local_importance_matrix,
                _CounterfactualExpV2SchemaConstants.SUMMARY_IMPORTANCE: summary_importance_list,
                _CounterfactualExpV2SchemaConstants.DATA_INTERFACE: data_interface,
                _CounterfactualExpV2SchemaConstants.FEATURE_NAMES: feature_names,
                _CounterfactualExpV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET: feature_names_including_target,
                _CounterfactualExpV2SchemaConstants.MODEL_TYPE: model_type,
                _CounterfactualExpV2SchemaConstants.DESIRED_CLASS: desired_class,
                _CounterfactualExpV2SchemaConstants.DESIRED_RANGE: desired_range,
                _CounterfactualExpV1SchemaConstants.METADATA: self.metadata
            }
            CounterfactualExplanations._check_cf_exp_output_against_json_schema(
                entire_dict, version=serialization_version)
            return json.dumps(entire_dict)
        else:
            raise UserConfigValidationException(
                "Unsupported serialization version {}".format(serialization_version))

    @staticmethod
    def _validate_serialization_version(version):
        if version is None:
            raise UserConfigValidationException("No version field in the json input")
        elif not _check_supported_json_output_versions(version):
            raise UserConfigValidationException("Incompatible version {} found in json input".format(version))

    @staticmethod
    def from_json(json_str):
        """ Deserialize json string to a CounterfactualExplanations object.
        """
        json_dict = json.loads(json_str)
        if _CommonSchemaConstants.METADATA in json_dict:
            version = json_dict[_CommonSchemaConstants.METADATA].get('version')
            CounterfactualExplanations._validate_serialization_version(version)

            if version == _SchemaVersions.V1:
                CounterfactualExplanations._check_cf_exp_output_against_json_schema(
                    json_dict, version=version)
                cf_examples_list = []
                for cf_examples_str in json_dict[_CounterfactualExpV1SchemaConstants.CF_EXAMPLES_LIST]:
                    cf_examples_list.append(CounterfactualExamples.from_json(cf_examples_str))

                return CounterfactualExplanations(
                        cf_examples_list=cf_examples_list,
                        local_importance=json_dict[_CounterfactualExpV1SchemaConstants.LOCAL_IMPORTANCE],
                        summary_importance=json_dict[_CounterfactualExpV1SchemaConstants.SUMMARY_IMPORTANCE],
                        version=version)
            else:
                CounterfactualExplanations._check_cf_exp_output_against_json_schema(
                    json_dict, version=version)
                cf_examples_list = []
                for index in range(0, len(json_dict[_CounterfactualExpV2SchemaConstants.CFS_LIST])):
                    # We need to save the json again since we need to recompose the
                    # counterfactual example.
                    cf_examples_str = json.dumps(
                        {
                            _DiverseCFV2SchemaConstants.FINAL_CFS_LIST: json_dict[
                                _CounterfactualExpV2SchemaConstants.CFS_LIST][index],
                            _DiverseCFV2SchemaConstants.TEST_INSTANCE_LIST: json_dict[
                                _CounterfactualExpV2SchemaConstants.TEST_DATA][index],
                            _DiverseCFV2SchemaConstants.DATA_INTERFACE: json_dict[
                                _CounterfactualExpV2SchemaConstants.DATA_INTERFACE],
                            _DiverseCFV2SchemaConstants.DESIRED_CLASS: json_dict[
                                _CounterfactualExpV2SchemaConstants.DESIRED_CLASS],
                            _DiverseCFV2SchemaConstants.DESIRED_RANGE: json_dict[
                                _CounterfactualExpV2SchemaConstants.DESIRED_RANGE],
                            _DiverseCFV2SchemaConstants.MODEL_TYPE: json_dict[
                                _CounterfactualExpV2SchemaConstants.MODEL_TYPE],
                            _DiverseCFV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET: json_dict[
                                _CounterfactualExpV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET]
                        }
                    )
                    cf_examples_list.append(
                        CounterfactualExamples.from_json(cf_examples_str)
                    )

                local_importance_list = None
                if json_dict[_CounterfactualExpV2SchemaConstants.LOCAL_IMPORTANCE] is not None:
                    local_importance_list = []
                    for local_importance_instance in json_dict[
                            _CounterfactualExpV2SchemaConstants.LOCAL_IMPORTANCE]:
                        local_importance_dict = {}
                        feature_names = json_dict[_CounterfactualExpV2SchemaConstants.FEATURE_NAMES]
                        for index in range(0, len(local_importance_instance)):
                            local_importance_dict[feature_names[index]] = local_importance_instance[index]
                        local_importance_list.append(local_importance_dict)

                summary_importance_dict = None
                if json_dict[_CounterfactualExpV2SchemaConstants.SUMMARY_IMPORTANCE] is not None:
                    summary_importance_dict = {}
                    feature_names = json_dict[
                        _CounterfactualExpV2SchemaConstants.FEATURE_NAMES]
                    for index in range(0, len(json_dict[
                            _CounterfactualExpV2SchemaConstants.SUMMARY_IMPORTANCE])):
                        summary_importance_dict[feature_names[index]] = json_dict[
                            _CounterfactualExpV2SchemaConstants.SUMMARY_IMPORTANCE][index]

                return CounterfactualExplanations(
                        cf_examples_list=cf_examples_list,
                        local_importance=local_importance_list,
                        summary_importance=summary_importance_dict,
                        version=version)
        else:
            return json_dict
