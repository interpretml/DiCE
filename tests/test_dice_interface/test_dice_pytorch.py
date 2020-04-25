import pytest

import dice_ml
from dice_ml.utils import helpers
import numpy as np

torch = pytest.importorskip("torch")

@pytest.fixture
def pyt_exp_object():
    backend = 'PYT'
    dataset = helpers.load_adult_income_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    exp = dice_ml.Dice(d, m)
    return exp

@pytest.fixture
def query_instance():
    query_instance = {'age':22,
          'workclass':'Private',
          'education':'HS-grad',
          'marital_status':'Single',
          'occupation':'Service',
          'race': 'White',
          'gender':'Female',
          'hours_per_week': 45}
    return query_instance

class TestDiceMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, pyt_exp_object, query_instance):
        self.exp = pyt_exp_object
        self.exp.do_cf_initializations(total_CFs=4, algorithm="DiverseCF", features_to_vary="all")
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        self.query_instance = query_instance.iloc[0].values
        self.exp.initialize_CFs(self.query_instance, init_near_query_instance=True)
        self.exp.target_cf_class = torch.tensor(1).float()
        np.random.seed(42)
        weights = np.random.rand(len(self.exp.data_interface.encoded_feature_names))
        self.exp.feature_weights_list = torch.tensor(weights)

    @pytest.mark.parametrize("yloss, output",[("hinge_loss", "4.7753"), ("l2_loss", "0.9999"), ("log_loss", "4.2892")])
    def test_first_loss(self, yloss, output):
        self.exp.yloss_type = yloss
        loss1 = self.exp.compute_first_part_of_loss()
        assert str(np.round(loss1.data.detach().numpy(),4)) == output

    def test_second_loss(self):
        self.exp.x1 =  torch.tensor(self.query_instance)
        loss2 = self.exp.compute_second_part_of_loss()
        assert str(np.round(loss2.data.detach().numpy(),4)) == "0.0068"

    @pytest.mark.parametrize("diversity_loss, output",[("dpp_style:inverse_dist", "0.0104"), ("avg_dist", "0.1743")])
    def test_third_loss(self, diversity_loss, output):
        self.exp.diversity_loss_type = diversity_loss
        loss3 = self.exp.compute_third_part_of_loss()
        assert str(np.round(loss3.data.detach().numpy(),4)) == output

    def test_fourth_loss(self):
        loss4 = self.exp.compute_fourth_part_of_loss()
        assert str(np.round(loss4.data.detach().numpy(),4)) == "0.2086"

    def test_final_cfs_and_preds(self, query_instance):
        dice_exp = self.exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
        test_cfs = [[57.0, 'Private', 'Doctorate', 'Married', 'White-Collar', 'White', 'Female', 46.0, 0.993], [33.0, 'Private', 'Prof-school', 'Married', 'Service', 'White', 'Male', 39.0, 0.964], [21.0, 'Self-Employed', 'Prof-school', 'Married', 'Service', 'White', 'Female', 47.0, 0.733], [49.0, 'Private', 'Masters', 'Married', 'Service', 'White', 'Female', 62.0, 0.957]]
        assert dice_exp.final_cfs_list == test_cfs

        preds = [np.round(preds.flatten().tolist(), 3)[0] for preds in dice_exp.final_cfs_preds]
        assert preds == [0.993, 0.964, 0.733, 0.957]
