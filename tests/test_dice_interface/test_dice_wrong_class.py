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


class _TieBoundaryClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, df):
        probs = []
        for val in df["x1"].to_numpy(dtype=float):
            if val <= 5:
                probs.append([0.9, 0.1])
            elif val <= 8:
                probs.append([0.5, 0.5])
            else:
                probs.append([0.2, 0.8])
        return np.array(probs, dtype=float)

    def predict(self, df):
        return self.classes_[np.argmax(self.predict_proba(df), axis=1)]


class TestWrongClassCounterfactuals:
    @classmethod
    def setup_class(cls):
        cls._stubbed_modules = _install_missing_dependency_stubs()

    @classmethod
    def teardown_class(cls):
        for mod_name in reversed(cls._stubbed_modules):
            sys.modules.pop(mod_name, None)

    def _build_random_explainer(self):
        import dice_ml

        df = pd.DataFrame({
            "x1": [1, 2, 3, 6, 7, 9, 10, 11],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "outcome": [0, 0, 0, 0, 0, 1, 1, 1],
        })
        data_interface = dice_ml.Data(
            dataframe=df,
            continuous_features=["x1"],
            outcome_name="outcome",
        )
        model_interface = dice_ml.Model(
            model=_TieBoundaryClassifier(),
            backend="sklearn",
            model_type="classifier",
        )
        return dice_ml.Dice(data_interface, model_interface, method="random")

    def test_random_rejects_boundary_tie_for_target_class_one(self):
        exp = self._build_random_explainer()
        query = pd.DataFrame({"x1": [2], "cat": ["a"]})
        res = exp.generate_counterfactuals(
            query,
            total_CFs=3,
            desired_class="opposite",
            sample_size=200,
            random_seed=42,
        )
        cf_example = res.cf_examples_list[0]
        assert (cf_example.final_cfs_df["outcome"] == 1).all()
        assert (cf_example.final_cfs_df_sparse["outcome"] == 1).all()

    def test_is_cf_valid_and_decide_cf_validity_tie_handling(self):
        exp = self._build_random_explainer()
        exp.num_output_nodes = 2
        exp.stopping_threshold = 0.5

        exp.target_cf_class = 1
        assert exp.decide_cf_validity(np.array([[0.5, 0.5]])).tolist() == [0]
        assert exp.is_cf_valid(np.array([0.5, 0.5])) is False
        assert exp.is_cf_valid(np.array([0.5])) is False
        assert exp.is_cf_valid(np.array([0.49, 0.51])) is True

        exp.target_cf_class = 0
        assert exp.decide_cf_validity(np.array([[0.5, 0.5]])).tolist() == [1]
        assert exp.is_cf_valid(np.array([0.5, 0.5])) is True

    def test_binary_search_preserves_valid_class(self):
        exp = self._build_random_explainer()
        exp.num_output_nodes = 2
        exp.target_cf_class = 0
        exp.stopping_threshold = 0.5

        # x1 <= 9 predicts class 0 (valid CF), x1 = 10 predicts class 1 (query instance).
        # Binary search midpoint round(9.5) = 10 hits right endpoint and must revert to 9.
        class _StepClassifier:
            classes_ = np.array([0, 1])

            def predict_proba(self, df):
                return np.array(
                    [[0.9, 0.1] if val <= 9 else [0.1, 0.9] for val in df["x1"].to_numpy(dtype=float)],
                    dtype=float,
                )

        exp.model.model = _StepClassifier()
        query = pd.DataFrame({"x1": [10], "cat": ["a"]})
        cf_sparse = pd.DataFrame({"x1": [9], "cat": ["a"], "outcome": [0]})
        res_sparse = exp.do_posthoc_sparsity_enhancement(
            cf_sparse.copy(),
            query,
            posthoc_sparsity_param=1.0,
            posthoc_sparsity_algorithm="binary",
            limit_steps_ls=100,
        )
        assert res_sparse["outcome"].iat[0] == 0
        assert res_sparse["x1"].iat[0] == 9
