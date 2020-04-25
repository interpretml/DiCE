import pytest

import dice_ml
from dice_ml.utils import helpers
import numpy as np

tf = pytest.importorskip("tensorflow")

@pytest.fixture
def tf_exp_object():
    backend = 'TF'+tf.__version__[0]
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
    def _initiate_exp_object(self, tf_exp_object, query_instance):
        self.exp = tf_exp_object
        self.exp.do_cf_initializations(total_CFs=4, algorithm="DiverseCF", features_to_vary="all")
        query_instance = self.exp.data_interface.prepare_query_instance(query_instance=query_instance, encode=True)
        self.query_instance = np.array([query_instance.iloc[0].values], dtype=np.float32)
        init_arrs = self.exp.initialize_CFs(self.query_instance, init_near_query_instance=True)
        np.random.seed(42)
        weights = np.random.rand(len(self.exp.data_interface.encoded_feature_names))
        weights = np.array([weights], dtype=np.float32)
        if tf.__version__[0] == '1':
            for i in range(4):
                self.exp.dice_sess.run(self.exp.cf_assign[i], feed_dict={self.exp.cf_init: init_arrs[i]})
            self.exp.feature_weights = tf.Variable(self.exp.minx, dtype=tf.float32)
            self.exp.dice_sess.run(tf.assign(self.exp.feature_weights, weights))
        else:
            self.exp.feature_weights_list = tf.constant([weights], dtype=tf.float32)

    @pytest.mark.parametrize("yloss, output",[("hinge_loss", "4.6711"), ("l2_loss", "[[0.9501]]"), ("log_loss", "[[3.6968]]")])
    def test_first_loss(self, yloss, output):
        if tf.__version__[0] == '1':
            loss1 = self.exp.compute_first_part_of_loss(method=yloss)
            loss1 = self.exp.dice_sess.run(loss1, feed_dict={self.exp.target_cf: np.array([[1]])})
        else:
            self.exp.target_cf_class = np.array([[1]], dtype=np.float32)
            self.exp.yloss_type = yloss
            loss1 = self.exp.compute_first_part_of_loss()
        assert str(np.round(loss1, 4)) == output

    def test_second_loss(self):
        if tf.__version__[0] == '1':
            loss2 = self.exp.compute_second_part_of_loss()
            loss2 = self.exp.dice_sess.run(loss2, feed_dict={self.exp.x1: self.query_instance})
        else:
            self.exp.x1 = tf.constant(self.query_instance, dtype=tf.float32)
            loss2 = self.exp.compute_second_part_of_loss()
        assert str(np.round(loss2, 4)) == '0.0068'

    @pytest.mark.parametrize("diversity_loss, output",[("dpp_style:inverse_dist", "0.0104"), ("avg_dist", "0.1743")])
    def test_third_loss(self, diversity_loss, output):
        if tf.__version__[0] == '1':
            loss3 = self.exp.compute_third_part_of_loss(diversity_loss)
            loss3 = self.exp.dice_sess.run(loss3)
        else:
            self.exp.diversity_loss_type = diversity_loss
            loss3 = self.exp.compute_third_part_of_loss()
        assert str(np.round(loss3, 4)) == output

    def test_fourth_loss(self):
        loss4 = self.exp.compute_fourth_part_of_loss()
        if tf.__version__[0] == '1':
            loss4 = self.exp.dice_sess.run(loss4)
        assert str(np.round(loss4, 4)) == '0.2086'

    def test_final_cfs_and_preds(self, query_instance):
        dice_exp = self.exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
        test_cfs = [[70.0, 'Private', 'Masters', 'Single', 'White-Collar', 'White', 'Female', 51.0, 0.534], [19.0, 'Self-Employed', 'Doctorate', 'Married', 'Service', 'White', 'Female', 44.0, 0.815], [47.0, 'Private', 'HS-grad', 'Married', 'Service', 'White', 'Female', 45.0, 0.589], [36.0, 'Private', 'Prof-school', 'Married', 'Service', 'White', 'Female', 62.0, 0.937]]
        assert dice_exp.final_cfs_list == test_cfs

        preds = [str(np.round(preds.flatten().tolist(), 3)[0]) for preds in dice_exp.final_cfs_preds]
        assert preds == ['0.534', '0.815', '0.589', '0.937']
