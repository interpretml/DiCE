import pytest

import numpy as np
import dice_ml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from dice_ml.utils.exception import SystemException


class TestModelClassification:

    def create_sklearn_random_forest_classifier(self, X, y):
        rfc = RandomForestClassifier(n_estimators=10, max_depth=4,
                                     random_state=777)
        model = rfc.fit(X, y)
        return model

    def test_base_model_classification(self, create_iris_data):
        x_train, x_test, y_train, y_test, feature_names, classes = \
            create_iris_data
        trained_model = self.create_sklearn_random_forest_classifier(x_train, y_train)

        diceml_model = dice_ml.Model(model=trained_model, backend='sklearn')
        diceml_model.transformer.initialize_transform_func()

        assert diceml_model is not None

        prediction_probabilities = diceml_model.get_output(x_test)
        assert prediction_probabilities.shape[0] == x_test.shape[0]
        assert prediction_probabilities.shape[1] == len(classes)

        predictions = diceml_model.get_output(x_test, model_score=False).reshape(-1, 1)
        assert predictions.shape[0] == x_test.shape[0]
        assert predictions.shape[1] == 1
        assert np.all(np.unique(predictions) == np.unique(y_test))

        with pytest.raises(NotImplementedError):
            diceml_model.get_gradient()

        assert diceml_model.get_num_output_nodes2(x_test) == len(classes)


class TestModelRegression:

    def create_sklearn_random_forest_regressor(self, X, y):
        rfc = RandomForestRegressor(n_estimators=10, max_depth=4,
                                    random_state=777)
        model = rfc.fit(X, y)
        return model

    def test_base_model_regression(self, create_boston_data):
        x_train, x_test, y_train, y_test, feature_names = \
            create_boston_data
        trained_model = self.create_sklearn_random_forest_regressor(x_train, y_train)

        diceml_model = dice_ml.Model(model=trained_model, model_type='regressor', backend='sklearn')
        diceml_model.transformer.initialize_transform_func()

        assert diceml_model is not None

        prediction_probabilities = diceml_model.get_output(x_test).reshape(-1, 1)
        assert prediction_probabilities.shape[0] == x_test.shape[0]
        assert prediction_probabilities.shape[1] == 1

        predictions = diceml_model.get_output(x_test, model_score=False).reshape(-1, 1)
        assert predictions.shape[0] == x_test.shape[0]
        assert predictions.shape[1] == 1

        with pytest.raises(NotImplementedError):
            diceml_model.get_gradient()

        with pytest.raises(SystemException):
            diceml_model.get_num_output_nodes2(x_test)
