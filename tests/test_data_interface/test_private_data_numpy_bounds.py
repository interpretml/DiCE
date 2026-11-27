import importlib.util
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


def _stub_raiutils():
    if importlib.util.find_spec("raiutils") is not None:
        return []
    raiutils_mod = types.ModuleType("raiutils")
    exceptions_mod = types.ModuleType("raiutils.exceptions")

    class UserConfigValidationException(Exception):
        pass

    exceptions_mod.UserConfigValidationException = UserConfigValidationException
    raiutils_mod.exceptions = exceptions_mod
    sys.modules["raiutils"] = raiutils_mod
    sys.modules["raiutils.exceptions"] = exceptions_mod
    return ["raiutils", "raiutils.exceptions"]


def _stub_jsonschema():
    if importlib.util.find_spec("jsonschema") is not None:
        return []
    jsonschema_mod = types.ModuleType("jsonschema")
    jsonschema_mod.validate = MagicMock()
    sys.modules["jsonschema"] = jsonschema_mod
    return ["jsonschema"]


def _stub_sklearn():
    if importlib.util.find_spec("sklearn") is not None:
        return []
    sklearn_mod = types.ModuleType("sklearn")
    preproc_mod = types.ModuleType("sklearn.preprocessing")
    neighbors_mod = types.ModuleType("sklearn.neighbors")
    compose_mod = types.ModuleType("sklearn.compose")
    ensemble_mod = types.ModuleType("sklearn.ensemble")
    model_sel_mod = types.ModuleType("sklearn.model_selection")
    pipeline_mod = types.ModuleType("sklearn.pipeline")

    class LabelEncoder:
        def fit(self, y):
            self.classes_ = np.unique(list(y))
            return self

        def transform(self, y):
            mapping = {val: idx for idx, val in enumerate(self.classes_)}
            return np.array([mapping[val] for val in y])

        def inverse_transform(self, y):
            return np.array([self.classes_[int(idx)] for idx in y])

    class FunctionTransformer:
        def __init__(self, func=None, inverse_func=None, validate=False,
                     check_inverse=True, kw_args=None, inv_kw_args=None):
            self.func = func
            self.inverse_func = inverse_func
            self.kw_args = kw_args or {}
            self.inv_kw_args = inv_kw_args or {}

        def transform(self, X):
            return X if self.func is None else self.func(X, **self.kw_args)

        def inverse_transform(self, X):
            return X if self.inverse_func is None else self.inverse_func(X, **self.inv_kw_args)

    preproc_mod.LabelEncoder = LabelEncoder
    preproc_mod.FunctionTransformer = FunctionTransformer
    preproc_mod.OneHotEncoder = MagicMock()
    neighbors_mod.KDTree = MagicMock()
    compose_mod.ColumnTransformer = MagicMock()
    ensemble_mod.RandomForestClassifier = MagicMock()
    model_sel_mod.train_test_split = MagicMock()
    pipeline_mod.Pipeline = MagicMock()

    sklearn_mod.preprocessing = preproc_mod
    sklearn_mod.neighbors = neighbors_mod
    sklearn_mod.compose = compose_mod
    sklearn_mod.ensemble = ensemble_mod
    sklearn_mod.model_selection = model_sel_mod
    sklearn_mod.pipeline = pipeline_mod

    modules = {
        "sklearn": sklearn_mod,
        "sklearn.preprocessing": preproc_mod,
        "sklearn.neighbors": neighbors_mod,
        "sklearn.compose": compose_mod,
        "sklearn.ensemble": ensemble_mod,
        "sklearn.model_selection": model_sel_mod,
        "sklearn.pipeline": pipeline_mod,
    }
    sys.modules.update(modules)
    return list(modules)


def _install_missing_dependency_stubs():
    """Install minimal module stubs only when optional runtime deps are absent."""
    return _stub_raiutils() + _stub_jsonschema() + _stub_sklearn()


class TestPrivateDataNumpyBounds:
    @classmethod
    def setup_class(cls):
        cls._stubbed_modules = _install_missing_dependency_stubs()

    @classmethod
    def teardown_class(cls):
        for mod_name in reversed(cls._stubbed_modules):
            sys.modules.pop(mod_name, None)

    def test_numpy_scalar_bounds_recognized_as_continuous(self):
        import dice_ml
        from dice_ml.explainer_interfaces.dice_random import DiceRandom

        df = pd.DataFrame({
            "int64_feat": pd.Series([1, 10], dtype="int64"),
            "int32_feat": pd.Series([0, 5], dtype="int32"),
            "float32_feat": pd.Series([1.0, 10.0], dtype="float32"),
            "float16_feat": pd.Series([2.0, 8.0], dtype="float16"),
        })
        data_ranges = {col: [df[col].min(), df[col].max()] for col in df.columns}
        data_ranges["cat_feat"] = ["a", "b"]

        data_interface = dice_ml.Data(features=data_ranges, outcome_name="outcome")

        assert data_interface.continuous_feature_names == [
            "int64_feat",
            "int32_feat",
            "float32_feat",
            "float16_feat",
        ]
        assert data_interface.categorical_feature_names == ["cat_feat"]

        model_interface = MagicMock()
        model_interface.model_type = "classifier"
        explainer = DiceRandom(data_interface, model_interface)

        query_df = pd.DataFrame({
            "int64_feat": [np.float16(3.5)],
            "int32_feat": [np.float32(2.5)],
            "float32_feat": [np.float16(4.5)],
            "float16_feat": [5.5],
            "cat_feat": ["a"],
        })
        features_to_vary = explainer.setup("all", None, query_df, "inverse_mad")
        assert features_to_vary == data_interface.feature_names

    def test_boolean_values_treated_as_categorical(self):
        import dice_ml

        data_interface = dice_ml.Data(
            features={
                "py_bool": [False, True],
                "np_bool": [np.bool_(False), np.bool_(True)],
                "num": [0, 1],
            },
            outcome_name="outcome",
        )
        assert data_interface.continuous_feature_names == ["num"]
        assert data_interface.categorical_feature_names == ["py_bool", "np_bool"]
