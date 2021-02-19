import numpy as np
import pandas as pd
import copy
from IPython.display import display

class CounterfactualExamples:
    """A class to store and visualize the resulting counterfactual explanations."""


    def __init__(self, data_interface, final_cfs_df, test_instance_df, final_cfs_df_sparse, posthoc_sparsity_param=0, desired_range=None, desired_class="opposite", encoding='one-hot', model_type='classifier'):

        self.data_interface = data_interface
        self.final_cfs_df = final_cfs_df
        self.test_instance_df = test_instance_df
        self.final_cfs_df_sparse = final_cfs_df_sparse

        self.final_cfs_list = None
        self.posthoc_sparsity_param = posthoc_sparsity_param # might be useful for future additions

        self.test_pred = self.test_instance_df[self.data_interface.outcome_name].iloc[0]
        if model_type == 'classifier':
            if desired_class == "opposite":
                self.new_outcome = 1.0 - round(self.test_pred)
            else:
                self.new_outcome = desired_class
        elif model_type == 'regressor':
            self.new_outcome = desired_range

        self.encoding = encoding

    def visualize_as_dataframe(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        display(self.test_instance_df) #  works only in Jupyter notebook
        if len(self.final_cfs_df) > 0:
            if self.posthoc_sparsity_param == None:
                print('\nCounterfactual set (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is not None:
                # CFs
                print('\nDiverse Counterfactual set (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df_sparse, show_only_changes)


            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is None:
                print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.display_df(self.final_cfs_df, show_only_changes)

            elif 'data_df' not in self.data_interface.__dict__: # for private data
                print('\nDiverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)

            else:
                # CFs
                print('\nDiverse Counterfactual set without sparsity correction (new outcome: ', self.new_outcome)
                self.display_df(self.final_cfs_df, show_only_changes)
        else:
            print('\nNo counterfactuals found!')

    def display_df(self, df, show_only_changes):
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
            display(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

    def visualize_as_list(self, display_sparse_df=True, show_only_changes=False):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        print(self.test_instance_df.values.tolist()[0])

        if len(self.final_cfs) > 0:
            if self.posthoc_sparsity_param == None:
                print('\nCounterfactual set (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is not None:
                # CFs
                print('\nDiverse Counterfactual set (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df_sparse.values.tolist(), show_only_changes)

            elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_df_sparse is None:
                print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            elif 'data_df' not in self.data_interface.__dict__: # for private data
                print('\nDiverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)

            else:
                # CFs
                print('\nDiverse Counterfactual set without sparsity correction (new outcome : %i)' %(self.new_outcome))
                self.print_list(self.final_cfs_df.values.tolist(), show_only_changes)
        else:
            print('\n0 counterfactuals found!')

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
