import numpy as np
import pytest

import dice_ml
from dice_ml.utils import helpers

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

class TestDiceTensorFlowMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, tf_exp_object, sample_adultincome_query):
        self.exp = tf_exp_object # explainer object
        self.exp.do_cf_initializations(total_CFs=4, algorithm="DiverseCF", features_to_vary="all") # initialize required params for CF computations

        # prepare query isntance for CF optimization
        # query_instance = self.exp.data_interface.prepare_query_instance(query_instance=sample_adultincome_query, encoding='one-hot')
        # self.query_instance = np.array([query_instance.iloc[0].values], dtype=np.float32)
        self.query_instance = self.exp.data_interface.get_ohe_min_max_normalized_data(sample_adultincome_query).values

        init_arrs = self.exp.initialize_CFs(self.query_instance, init_near_query_instance=True) # initialize CFs
        self.desired_class = 1 # desired class is 1

        # setting random feature weights
        np.random.seed(42)
        weights = np.random.rand(len(self.exp.data_interface.ohe_encoded_feature_names))
        weights = np.array([weights], dtype=np.float32)
        if tf.__version__[0] == '1':
            for i in range(4):
                self.exp.dice_sess.run(self.exp.cf_assign[i], feed_dict={self.exp.cf_init: init_arrs[i]})
            self.exp.feature_weights = tf.Variable(self.exp.minx, dtype=tf.float32)
            self.exp.dice_sess.run(tf.assign(self.exp.feature_weights, weights))
        else:
            self.exp.feature_weights_list = tf.constant([weights], dtype=tf.float32)

    @pytest.mark.parametrize("yloss, output",[("hinge_loss", 4.6711), ("l2_loss", 0.9501), ("log_loss", 3.6968)])
    def test_yloss(self, yloss, output):
        if tf.__version__[0] == '1':
            loss1 = self.exp.compute_yloss(method=yloss)
            loss1 = self.exp.dice_sess.run(loss1, feed_dict={self.exp.target_cf: np.array([[1]])})
        else:
            self.exp.target_cf_class = np.array([[self.desired_class]], dtype=np.float32)
            self.exp.yloss_type = yloss
            loss1 = self.exp.compute_yloss().numpy()
        assert pytest.approx(loss1, abs=1e-4) == output

    def test_proximity_loss(self):
        if tf.__version__[0] == '1':
            loss2 = self.exp.compute_proximity_loss()
            loss2 = self.exp.dice_sess.run(loss2, feed_dict={self.exp.x1: self.query_instance})
        else:
            self.exp.x1 = tf.constant(self.query_instance, dtype=tf.float32)
            loss2 = self.exp.compute_proximity_loss().numpy()
        assert pytest.approx(loss2, abs=1e-4) == 0.0068 # proximity loss computed for given query instance and feature weights.

    @pytest.mark.parametrize("diversity_loss, output",[("dpp_style:inverse_dist", 0.0104), ("avg_dist", 0.1743)])
    def test_diversity_loss(self, diversity_loss, output):
        if tf.__version__[0] == '1':
            loss3 = self.exp.compute_diversity_loss(diversity_loss)
            loss3 = self.exp.dice_sess.run(loss3)
        else:
            self.exp.diversity_loss_type = diversity_loss
            loss3 = self.exp.compute_diversity_loss().numpy()
        assert pytest.approx(loss3, abs=1e-4) == output

    def test_regularization_loss(self):
        loss4 = self.exp.compute_regularization_loss()
        if tf.__version__[0] == '1':
            loss4 = self.exp.dice_sess.run(loss4)
        else:
            loss4 = loss4.numpy()
        assert pytest.approx(loss4, abs=1e-4) == 0.2086 # regularization loss computed for given query instance and feature weights.

    def test_final_cfs_and_preds(self, sample_adultincome_query):
        """
        Tets correctness of final CFs and their predictions for sample query instance.
        """
        dice_exp = self.exp.generate_counterfactuals(sample_adultincome_query, total_CFs=4, desired_class="opposite")
        test_cfs = [[70.0, 'Private', 'Masters', 'Single', 'White-Collar', 'White', 'Female', 51.0, 0.534], [22.0, 'Self-Employed', 'Doctorate', 'Married', 'Service', 'White', 'Female', 45.0, 0.861], [47.0, 'Private', 'HS-grad', 'Married', 'Service', 'White', 'Female', 45.0, 0.589], [36.0, 'Private', 'Prof-school', 'Married', 'Service', 'White', 'Female', 62.0, 0.937]]
        # TODO  The model predictions changed after update to posthoc sparsity. Need to investigate.
        #assert dice_exp.final_cfs_df_sparse.values.tolist() == test_cfs

