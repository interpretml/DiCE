import numpy as np
import pytest

import dice_ml
from dice_ml.utils import helpers


@pytest.fixture
def KD_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='kdtree')
    return exp


class TestDiceKDBinaryClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, KD_binary_classification_exp_object):
        self.exp = KD_binary_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    # Verifying the output of the KD tree
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    @pytest.mark.parametrize('posthoc_sparsity_algorithm', ['linear', 'binary', None])
    def test_KD_tree_output(self, desired_class, sample_custom_query_1, total_CFs, posthoc_sparsity_algorithm):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_1, desired_class=desired_class,
                                           total_CFs=total_CFs,
                                           posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df

        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[0]) and \
               all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[0])

    # Testing that the permitted_range argument actually varies the features only within the permitted_range
    @pytest.mark.parametrize("desired_class, total_CFs, permitted_range", [(0, 1, {'Numerical': [1000, 10000]})])
    def test_permitted_range(self, desired_class, sample_custom_query_2, total_CFs, permitted_range):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_2, desired_class=desired_class,
                                           total_CFs=total_CFs, permitted_range=permitted_range)
        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        expected_output = self.exp.data_interface.data_df
        assert all(self.exp.final_cfs_df.Numerical == expected_output.Numerical[1]) and \
               all(self.exp.final_cfs_df.Categorical == expected_output.Categorical[1])

    # Ensuring that there are no duplicates in the resulting counterfactuals even if the dataset has duplicates
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 2)])
    def test_duplicates(self, desired_class, sample_custom_query_4, total_CFs):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_4, total_CFs=total_CFs,
                                           desired_class=desired_class)

        self.exp.final_cfs_df.Numerical = self.exp.final_cfs_df.Numerical.astype(int)
        self.exp.final_cfs_df = self.exp.final_cfs_df.reset_index(drop=True)

        expected_output = self.exp.data_interface.data_df.iloc[np.r_[2, 0]][self.exp.data_interface.feature_names]
        expected_output = expected_output.reset_index(drop=True)

        assert all(self.exp.final_cfs_df == expected_output)

    # Testing for index returned
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 1)])
    @pytest.mark.parametrize('posthoc_sparsity_algorithm', ['linear', 'binary', None])
    def test_index(self, desired_class, sample_custom_query_index, total_CFs, posthoc_sparsity_algorithm):
        self.exp._generate_counterfactuals(query_instance=sample_custom_query_index, total_CFs=total_CFs,
                                           desired_class=desired_class,
                                           posthoc_sparsity_algorithm=posthoc_sparsity_algorithm)
        assert self.exp.final_cfs_df.index[0] == 3
