
import numpy as np
import pandas as pd

class CounterfactualExamples:
    """A class to store and visualize the resulting counterfactual explanations."""

    def __init__(self, data_interface, test_instance, test_pred, final_cfs, final_cfs_preds, final_cfs_sparse, cfs_preds_sparse, posthoc_sparsity_param):

        self.data_interface = data_interface
        self.test_instance = test_instance
        self.test_pred = test_pred
        self.final_cfs = final_cfs
        self.final_cfs_preds = final_cfs_preds
        self.final_cfs_sparse = final_cfs_sparse
        self.cfs_preds_sparse = cfs_preds_sparse
        self.posthoc_sparsity_param = posthoc_sparsity_param

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
        k = self.data_interface.continuous_feature_names
        result = result.round(dict(zip(k,v)))

        # predictions for CFs
        test_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds_sparse]
        test_preds = [item for sublist in test_preds for item in sublist]
        test_preds = np.array(test_preds)

        result[self.data_interface.outcome_name] = test_preds
        self.final_cfs_df_sparse = result[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        self.final_cfs_list_sparse = self.final_cfs_df_sparse.values.tolist()

    def visualize_as_dataframe(self, display_sparse_df=True):

        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        display(self.org_instance) #  works only in Jupyter notebook

        if 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is not None:
            # CFs
            print('\nDiverse Counterfactual set after sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            display(self.final_cfs_df_sparse)  #  works only in Jupyter notebook

        elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is None:
            print('Please specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying without posthoc sparsity operation')
            display(self.final_cfs_df)  #  works only in Jupyter notebook
        else:
            # CFs
            print('\nDiverse Counterfactual set (new outcome : %i)' %(1-round(self.test_pred)))
            display(self.final_cfs_df)  #  works only in Jupyter notebook


    def visualize_as_list(self, display_sparse_df=True):
        # original instance
        print('Query instance (original outcome : %i)' %round(self.test_pred))
        print(self.org_instance.values.tolist()[0])

        if 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is not None:
            # CFs
            print('\nDiverse Counterfactual set after sparsity correction (new outcome : %i)' %(1-round(self.test_pred)))
            for ix in range(len(self.final_cfs_list_sparse)):
                print(self.final_cfs_list_sparse[ix])

        elif 'data_df' in self.data_interface.__dict__ and display_sparse_df==True and self.final_cfs_sparse is None:
            print('Please specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying without posthoc sparsity operation')
            for ix in range(len(self.final_cfs_list)):
                print(self.final_cfs_list[ix])
        else:
            # CFs
            print('\nDiverse Counterfactual set (new outcome : %i)' %(1-round(self.test_pred)))
            for ix in range(len(self.final_cfs_list)):
                print(self.final_cfs_list[ix])
