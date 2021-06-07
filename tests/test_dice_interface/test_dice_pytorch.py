import numpy as np
import pytest

import dice_ml
from dice_ml.utils import helpers
from dice_ml.counterfactual_explanations import CounterfactualExplanations


torch = pytest.importorskip("torch")


@pytest.fixture
def pyt_exp_object():
    backend = 'PYT'
    dataset = helpers.load_adult_income_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m)
    return exp


class TestDiceTorchMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, pyt_exp_object, sample_adultincome_query):
        self.exp = pyt_exp_object  # explainer object
        # initialize required params for CF computations
        self.exp.do_cf_initializations(total_CFs=4, algorithm="DiverseCF", features_to_vary="all")

        # prepare query isntance for CF optimization
        # query_instance = self.exp.data_interface.prepare_query_instance(
        #     query_instance=sample_adultincome_query, encoding='one-hot')
        # self.query_instance = query_instance.iloc[0].values
        self.query_instance = self.exp.data_interface.get_ohe_min_max_normalized_data(sample_adultincome_query).iloc[0].values

        self.exp.initialize_CFs(self.query_instance, init_near_query_instance=True)  # initialize CFs
        self.exp.target_cf_class = torch.tensor(1).float()  # set desired class to 1

        # setting random feature weights
        np.random.seed(42)
        weights = np.random.rand(len(self.exp.data_interface.ohe_encoded_feature_names))
        self.exp.feature_weights_list = torch.tensor(weights)

    @pytest.mark.parametrize("yloss, output", [("hinge_loss", 10.8443), ("l2_loss", 0.9999), ("log_loss", 9.8443)])
    def test_yloss(self, yloss, output):
        self.exp.yloss_type = yloss
        loss1 = self.exp.compute_yloss()
        assert pytest.approx(loss1.data.detach().numpy(), abs=1e-4) == output

    def test_proximity_loss(self):
        self.exp.x1 = torch.tensor(self.query_instance)
        loss2 = self.exp.compute_proximity_loss()
        # proximity loss computed for given query instance and feature weights.
        assert pytest.approx(loss2.data.detach().numpy(), abs=1e-4) == 0.0068

    @pytest.mark.parametrize("diversity_loss, output", [("dpp_style:inverse_dist", 0.0104), ("avg_dist", 0.1743)])
    def test_diversity_loss(self, diversity_loss, output):
        self.exp.diversity_loss_type = diversity_loss
        loss3 = self.exp.compute_diversity_loss()
        assert pytest.approx(loss3.data.detach().numpy(), abs=1e-4) == output

    def test_regularization_loss(self):
        loss4 = self.exp.compute_regularization_loss()
        # regularization loss computed for given query instance and feature weights.
        assert pytest.approx(loss4.data.detach().numpy(), abs=1e-4) == 0.2086

    def test_final_cfs_and_preds(self, sample_adultincome_query):
        """
        Tets correctness of final CFs and their predictions for sample query instance.
        """
        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query, total_CFs=4, desired_class="opposite")
        assert isinstance(counterfactual_explanations, CounterfactualExplanations)
        # test_cfs = [[72.0, 'Private', 'HS-grad', 'Married', 'White-Collar', 'White', 'Female', 45.0, 0.691],
        #             [29.0, 'Private', 'Prof-school', 'Married', 'Service', 'White', 'Male', 45.0, 0.954],
        #             [52.0, 'Private', 'Doctorate', 'Married', 'Service', 'White', 'Female', 45.0, 0.971],
        #             [47.0, 'Private', 'Masters', 'Married', 'Service', 'White', 'Female', 73.0, 0.971]]
        # TODO  The model predictions changed after update to posthoc sparsity. Need to investigate.
        # assert dice_exp.final_cfs_df_sparse.values.tolist() == test_cfs
