import pytest
import numpy as np
import dice_ml
from dice_ml.utils import helpers
from dice_ml.utils.exception import UserConfigValidationException
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.counterfactual_explanations import CounterfactualExplanations


@pytest.fixture
def random_binary_classification_exp_object():
    backend = 'sklearn'
    dataset = helpers.load_custom_testing_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['Numerical'], outcome_name='Outcome')
    ML_modelpath = helpers.get_custom_dataset_modelpath_pipeline()
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m, method='random')
    return exp


class TestDiceRandomBinaryClassificationMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, random_binary_classification_exp_object):
        self.exp = random_binary_classification_exp_object  # explainer object
        self.data_df_copy = self.exp.data_interface.data_df.copy()

    # Verifying the output of the random tree
    @pytest.mark.parametrize("desired_class, total_CFs", [(0, 2)])
    def test_random_counterfactual_explanations_output(self, desired_class, sample_custom_query_1, total_CFs):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            query_instances=sample_custom_query_1, desired_class=desired_class,
            total_CFs=total_CFs)

        assert counterfactual_explanations is not None
        assert len(counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
        json_output = counterfactual_explanations.to_json()
        assert json_output is not None

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(json_output)
        assert recovered_counterfactual_explanations is not None
        assert recovered_counterfactual_explanations == counterfactual_explanations

        assert len(recovered_counterfactual_explanations.cf_examples_list) == sample_custom_query_1.shape[0]
        assert recovered_counterfactual_explanations.cf_examples_list[0].final_cfs_df.shape[0] == total_CFs
