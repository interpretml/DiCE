import copy
import json

import pandas as pd

from dice_ml.constants import ModelTypes, _SchemaVersions
from dice_ml.utils.serialize import DummyDataInterface


class _DiverseCFV1SchemaConstants:
    DATA_INTERFACE = 'data_interface'
    MODEL_TYPE = 'model_type'
    DESIRED_CLASS = 'desired_class'
    DESIRED_RANGE = 'desired_range'
    TEST_INSTANCE_DF = 'test_instance_df'
    FINAL_CFS_DF = 'final_cfs_df'


class _DiverseCFV2SchemaConstants:
    DATA_INTERFACE = 'data_interface'
    MODEL_TYPE = 'model_type'
    DESIRED_CLASS = 'desired_class'
    DESIRED_RANGE = 'desired_range'
    FEATURE_NAMES_INCLUDING_TARGET = 'feature_names_including_target'
    FEATURE_NAMES = 'feature_names'
    TEST_INSTANCE_LIST = 'test_instance_list'
    FINAL_CFS_LIST = 'final_cfs_list'


def json_converter(obj):
    """ Helper function to convert object to json.
    """
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__


class CounterfactualExamples:
    """A class to store and visualize the resulting counterfactual explanations."""
    def __init__(self, data_interface=None, final_cfs_df=None, test_instance_df=None,
                 final_cfs_df_sparse=None, posthoc_sparsity_param=0,
                 desired_range=None, desired_class="opposite",
                 model_type=ModelTypes.Classifier):

        self.data_interface = data_interface
        self.final_cfs_df = final_cfs_df
        self.test_instance_df = test_instance_df
        self.final_cfs_df_sparse = final_cfs_df_sparse
        self.model_type = model_type
        self.desired_class = desired_class
        self.desired_range = desired_range

        self.final_cfs_list = None
        self.posthoc_sparsity_param = posthoc_sparsity_param  # might be useful for future additions

        self.test_pred = self.test_instance_df[self.data_interface.outcome_name].iat[0]
        if model_type == ModelTypes.Classifier:
            if desired_class == "opposite":
                self.new_outcome = 1.0 - round(self.test_pred)
            else:
                self.new_outcome = desired_class
        elif model_type == ModelTypes.Regressor:
            self.new_outcome = desired_range

    def __eq__(self, other_counterfactual_example):
        if isinstance(other_counterfactual_example, CounterfactualExamples):
            return self.desired_class == other_counterfactual_example.desired_class and \
                        self.desired_range == other_counterfactual_example.desired_range and \
                        self.model_type == other_counterfactual_example.model_type and \
                        (self.final_cfs_df is None) == \
                        (other_counterfactual_example.final_cfs_df is None) and \
                        (self.final_cfs_df_sparse is None) == \
                        (other_counterfactual_example.final_cfs_df_sparse is None)
        return False

    def _dump_output(self, content, show_only_changes=False, is_notebook_console=False):
        if is_notebook_console:
            self.display_df(content, show_only_changes=show_only_changes)
        else:
            assert isinstance(content, pd.DataFrame), "Expecting a pandas dataframe"
            self.print_list(content.values.tolist(),
                            show_only_changes=show_only_changes)

    def _visualize_internal(self, display_sparse_df=True, show_only_changes=False,
                            is_notebook_console=False):
        if self.final_cfs_df is not None and len(self.final_cfs_df) > 0:
            if self.posthoc_sparsity_param is None:
                print('\nCounterfactual set (new outcome: {0})'.format(self.new_outcome))
                self._dump_output(content=self.final_cfs_df, show_only_changes=show_only_changes,
                                  is_notebook_console=is_notebook_console)
            elif hasattr(self.data_interface, 'data_df') and \
                    display_sparse_df is True and self.final_cfs_df_sparse is not None:
                # CFs
                print('\nDiverse Counterfactual set (new outcome: {0})'.format(self.new_outcome))
                self._dump_output(content=self.final_cfs_df_sparse, show_only_changes=show_only_changes,
                                  is_notebook_console=is_notebook_console)
            elif hasattr(self.data_interface, 'data_df') and \
                    display_sparse_df is True and self.final_cfs_df_sparse is None:
                print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. ',
                      'displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %
                      (self.new_outcome))
                self._dump_output(content=self.final_cfs_df, show_only_changes=show_only_changes,
                                  is_notebook_console=is_notebook_console)
            elif not hasattr(self.data_interface, 'data_df'):  # for private data
                print('\nDiverse Counterfactual set without sparsity correction since only metadata about each',
                      ' feature is available (new outcome: %i)' % (self.new_outcome))
                self._dump_output(content=self.final_cfs_df, show_only_changes=show_only_changes,
                                  is_notebook_console=is_notebook_console)
            else:
                # CFs
                print('\nDiverse Counterfactual set without sparsity correction (new outcome: ', self.new_outcome)
                self._dump_output(content=self.final_cfs_df, show_only_changes=show_only_changes,
                                  is_notebook_console=is_notebook_console)
        else:
            print('\nNo counterfactuals found!')

    def visualize_as_dataframe(self, display_sparse_df=True, show_only_changes=False):
        from IPython.display import display

        # original instance
        print('Query instance (original outcome : %i)' % round(self.test_pred))
        display(self.test_instance_df)  # works only in Jupyter notebook
        self._visualize_internal(display_sparse_df=display_sparse_df,
                                 show_only_changes=show_only_changes,
                                 is_notebook_console=True)

    def display_df(self, df, show_only_changes):
        from IPython.display import display
        if show_only_changes is False:
            display(df)  # works only in Jupyter notebook
        else:
            newdf = df.values.tolist()
            org = self.test_instance_df.values.tolist()[0]
            for ix in range(df.shape[0]):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=df.columns, index=df.index))  # works only in Jupyter notebook

    def visualize_as_list(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' % round(self.test_pred))
        print(self.test_instance_df.values.tolist()[0])
        self._visualize_internal(display_sparse_df=display_sparse_df,
                                 show_only_changes=show_only_changes,
                                 is_notebook_console=False)

    def print_list(self, li, show_only_changes):
        if show_only_changes is False:
            for ix in range(len(li)):
                print(li[ix])
        else:
            newli = copy.deepcopy(li)
            org = self.test_instance_df.values.tolist()[0]
            for ix in range(len(newli)):
                for jx in range(len(newli[ix])):
                    if newli[ix][jx] == org[jx]:
                        newli[ix][jx] = '-'
                print(newli[ix])

    def to_json(self, serialization_version):
        if self.final_cfs_df_sparse is not None:
            df = self.final_cfs_df_sparse
        else:
            df = self.final_cfs_df

        dummy_data_interface = None
        if hasattr(self.data_interface, 'data_df'):
            dummy_data_interface = DummyDataInterface(
                    self.data_interface.outcome_name,
                    "dummy_data")
        else:
            dummy_data_interface = DummyDataInterface(
                    self.data_interface.outcome_name)

        if serialization_version == _SchemaVersions.V1:
            obj = {
                _DiverseCFV1SchemaConstants.DATA_INTERFACE: dummy_data_interface,
                _DiverseCFV1SchemaConstants.MODEL_TYPE: self.model_type,
                _DiverseCFV1SchemaConstants.DESIRED_CLASS: self.desired_class,
                _DiverseCFV1SchemaConstants.DESIRED_RANGE: self.desired_range,
                _DiverseCFV1SchemaConstants.TEST_INSTANCE_DF: self.test_instance_df,
                _DiverseCFV1SchemaConstants.FINAL_CFS_DF: df
            }
            return json.dumps(obj, default=json_converter)
        elif serialization_version == _SchemaVersions.V2:
            dummy_data_interface_dict = dummy_data_interface.to_json()
            feature_names_including_target = self.test_instance_df.columns.tolist()
            feature_names = self.test_instance_df.columns.tolist().copy()
            feature_names.remove(dummy_data_interface.outcome_name)
            test_instance_df_as_list = self.test_instance_df.values.tolist()
            if df is not None:
                final_cfs_df_as_as_list = df.values.tolist()
            else:
                final_cfs_df_as_as_list = None

            alternate_obj = {
                _DiverseCFV2SchemaConstants.TEST_INSTANCE_LIST: test_instance_df_as_list,
                _DiverseCFV2SchemaConstants.FINAL_CFS_LIST: final_cfs_df_as_as_list,
                _DiverseCFV2SchemaConstants.DATA_INTERFACE: dummy_data_interface_dict,
                _DiverseCFV2SchemaConstants.FEATURE_NAMES: feature_names,
                _DiverseCFV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET: feature_names_including_target,
                _DiverseCFV2SchemaConstants.MODEL_TYPE: self.model_type,
                _DiverseCFV2SchemaConstants.DESIRED_CLASS: self.desired_class,
                _DiverseCFV2SchemaConstants.DESIRED_RANGE: self.desired_range
            }
            return json.dumps(alternate_obj)

    @staticmethod
    def from_json(cf_example_json_str):
        cf_example_dict = json.loads(cf_example_json_str)
        if cf_example_dict.get(_DiverseCFV1SchemaConstants.TEST_INSTANCE_DF) is not None:
            test_instance_df = pd.read_json(cf_example_dict[
                _DiverseCFV1SchemaConstants.TEST_INSTANCE_DF])
            if cf_example_dict[_DiverseCFV1SchemaConstants.FINAL_CFS_DF] is not None:
                cfs_df = pd.read_json(cf_example_dict[_DiverseCFV1SchemaConstants.FINAL_CFS_DF])
            else:
                cfs_df = None

            # Creating the object for dummy_data_interface
            dummy_data_interface = DummyDataInterface(**cf_example_dict[_DiverseCFV1SchemaConstants.DATA_INTERFACE])
            return CounterfactualExamples(data_interface=dummy_data_interface,
                                          test_instance_df=test_instance_df,
                                          final_cfs_df=cfs_df,
                                          final_cfs_df_sparse=cfs_df,
                                          posthoc_sparsity_param=None,
                                          desired_class=cf_example_dict[_DiverseCFV1SchemaConstants.DESIRED_CLASS],
                                          desired_range=cf_example_dict[_DiverseCFV1SchemaConstants.DESIRED_RANGE],
                                          model_type=cf_example_dict[_DiverseCFV1SchemaConstants.MODEL_TYPE])
        else:
            final_cfs_list = cf_example_dict[_DiverseCFV2SchemaConstants.FINAL_CFS_LIST]
            test_instance_list = cf_example_dict[_DiverseCFV2SchemaConstants.TEST_INSTANCE_LIST]
            feature_names_including_target = cf_example_dict[_DiverseCFV2SchemaConstants.FEATURE_NAMES_INCLUDING_TARGET]

            data_interface = cf_example_dict[_DiverseCFV2SchemaConstants.DATA_INTERFACE]
            desired_class = cf_example_dict[_DiverseCFV2SchemaConstants.DESIRED_CLASS]
            desired_range = cf_example_dict[_DiverseCFV2SchemaConstants.DESIRED_RANGE]
            model_type = cf_example_dict[_DiverseCFV2SchemaConstants.MODEL_TYPE]

            test_instance_df = pd.DataFrame(data=test_instance_list,
                                            columns=feature_names_including_target)
            if final_cfs_list is not None:
                cfs_df = pd.DataFrame(data=final_cfs_list,
                                      columns=feature_names_including_target)
            else:
                cfs_df = None
            # Creating the object for dummy_data_interface
            dummy_data_interface = DummyDataInterface(**data_interface)
            return CounterfactualExamples(data_interface=dummy_data_interface,
                                          test_instance_df=test_instance_df,
                                          final_cfs_df=cfs_df,
                                          final_cfs_df_sparse=cfs_df,
                                          posthoc_sparsity_param=None,
                                          desired_class=desired_class,
                                          desired_range=desired_range,
                                          model_type=model_type)
