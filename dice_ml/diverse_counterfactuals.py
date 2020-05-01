import numpy as np
import pandas as pd
import copy
from IPython.display import display

class CounterfactualExamples:
    """A class to store and visualize the resulting counterfactual explanations."""

    def __init__(self, data_interface, test_instance, test_pred, final_cfs, final_cfs_preds, final_cfs_sparse=None, cfs_preds_sparse=None, posthoc_sparsity_param=0):

        self.data_interface = data_interface
        self.test_instance = test_instance
        self.test_pred = test_pred
        self.final_cfs = final_cfs
        self.final_cfs_preds = final_cfs_preds
        self.final_cfs_sparse = final_cfs_sparse
        self.cfs_preds_sparse = cfs_preds_sparse
        self.posthoc_sparsity_param = posthoc_sparsity_param # might be useful for future additions

        self.convert_to_dataframe() # transforming the test input from numpy to pandas dataframe
        if self.final_cfs_sparse is not None:
            self.convert_to_dataframe_sparse()


    def convert_to_dataframe(self):
        test_instance_updated = pd.DataFrame(np.array([np.append(self.test_instance, self.test_pred)]), columns = self.data_interface.encoded_feature_names+[self.data_interface.outcome_name])

        org_instance = self.data_interface.from_dummies(test_instance_updated)
        org_instance = org_instance[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        self.org_instance = self.data_interface.de_normalize_data(org_instance)

        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])

        result = self.data_interface.get_decoded_data(cfs)
        result = self.data_interface.de_normalize_data(result)

        v = self.data_interface.get_decimal_precisions()
        # k = self.data_interface.continuous_feature_names
        # result = result.round(dict(zip(k,v)))
        for ix, feature in enumerate(self.data_interface.continuous_feature_names):
            result[feature] = result[feature].astype(float).round(v[ix])

        # predictions for CFs
        test_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.final_cfs_preds]
        test_preds = [item for sublist in test_preds for item in sublist]
        test_preds = np.array(test_preds)

        result[self.data_interface.outcome_name] = test_preds
        self.final_cfs_df = result[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        self.final_cfs_list = self.final_cfs_df.values.tolist()

    def convert_to_dataframe_sparse(self):
        test_instance_updated = pd.DataFrame(np.array([np.append(self.test_instance, self.test_pred)]), columns = self.data_interface.encoded_feature_names+[self.data_interface.outcome_name])

        org_instance = self.data_interface.from_dummies(test_instance_updated)
        org_instance = org_instance[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        self.org_instance = self.data_interface.de_normalize_data(org_instance)

        cfs = np.array([self.final_cfs_sparse[i][0] for i in range(len(self.final_cfs_sparse))])

        result = self.data_interface.get_decoded_data(cfs)
        result = self.data_interface.de_normalize_data(result)

        v = self.data_interface.get_decimal_precisions()
        # k = self.data_interface.continuous_feature_names
        # result = result.round(dict(zip(k,v)))
        for ix, feature in enumerate(self.data_interface.continuous_feature_names):
            result[feature] = result[feature].astype(float).round(v[ix])

        # predictions for CFs
        test_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds_sparse]
        test_preds = [item for sublist in test_preds for item in sublist]
        test_preds = np.array(test_preds)

        result[self.data_interface.outcome_name] = test_preds
        self.final_cfs_df_sparse = result[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        self.final_cfs_list_sparse = self.final_cfs_df_sparse.values.tolist()

    def visualize_as_dataframe(self, display_sparse_df=True, show_only_changes=False):

        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        display(self.org_instance) #  works only in Jupyter notebook

        if 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is not None:
            # CFs
            print('\nDiverse Counterfactual set (new outcome : %i)' %(1-round(self.test_pred)))
            self.display_df(self.final_cfs_df_sparse, show_only_changes)

        elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is None:
            print('Please specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            self.display_df(self.final_cfs_df, show_only_changes)

        elif 'data_df' not in self.data_interface.__dict__: # for private data
            print('Diverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome : %i)' %(1-round(self.test_pred)))
            self.display_df(self.final_cfs_df, show_only_changes)

        else:
            # CFs
            print('\nDiverse Counterfactual set without sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            self.display_df(self.final_cfs_df, show_only_changes)

    def display_df(self, df, show_only_changes):
        if show_only_changes is False:
            display(df)  #  works only in Jupyter notebook
        else:
            newdf = df.values
            org = self.org_instance.values.tolist()[0]
            for ix in range(df.shape[0]):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=df.columns)) #  works only in Jupyter notebook

    def visualize_as_list(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        print(self.org_instance.values.tolist()[0])

        if 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is not None:
            # CFs
            print('\nDiverse Counterfactual set (new outcome : %i)' %(1-round(self.test_pred)))
            self.print_list(self.final_cfs_list_sparse, show_only_changes)

        elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is None:
            print('Please specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            self.print_list(self.final_cfs_list_sparse, show_only_changes)

        elif 'data_df' not in self.data_interface.__dict__: # for private data
            print('Diverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome : %i)' %(1-round(self.test_pred)))
            self.print_list(self.final_cfs_list, show_only_changes)

        else:
            # CFs
            print('\nDiverse Counterfactual set without sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            self.print_list(self.final_cfs_list, show_only_changes)

    def print_list(self, li, show_only_changes):
        if show_only_changes is False:
            for ix in range(len(li)):
                print(li[ix])
        else:
            newli = copy.deepcopy(li)
            org = self.org_instance.values.tolist()[0]
            for ix in range(len(newli)):
                for jx in range(len(newli[ix])):
                    if newli[ix][jx] == org[jx]:
                        newli[ix][jx] = '-'
                print(newli[ix])
