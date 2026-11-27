import importlib.util
import sys
import types
from unittest.mock import MagicMock

import numpy as np


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


def _stub_torch():
    if importlib.util.find_spec("torch") is not None:
        return []
    torch_mod = types.ModuleType("torch")
    torch_mod.utils = types.ModuleType("torch.utils")
    torch_mod.utils.data = types.ModuleType("torch.utils.data")
    sys.modules["torch"] = torch_mod
    sys.modules["torch.utils"] = torch_mod.utils
    sys.modules["torch.utils.data"] = torch_mod.utils.data
    return ["torch", "torch.utils", "torch.utils.data"]


def _install_missing_dependency_stubs():
    """Install minimal module stubs only when optional runtime deps are absent."""
    return _stub_raiutils() + _stub_jsonschema() + _stub_sklearn() + _stub_torch()


class TestDicePyTorchPrivateData:
    @classmethod
    def setup_class(cls):
        cls._stubbed_modules = _install_missing_dependency_stubs()

    @classmethod
    def teardown_class(cls):
        for mod_name in reversed(cls._stubbed_modules):
            sys.modules.pop(mod_name, None)

    def test_init_with_private_data_interface(self):
        import dice_ml
        from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
        from dice_ml.utils.helpers import DataTransfomer

        data_interface = dice_ml.Data(
            features={
                "age": [17, 90],
                "workclass": ["Government", "Private", "Self-Employed"],
                "hours_per_week": [1, 99],
            },
            outcome_name="income",
        )

        model_interface = MagicMock()
        model_interface.transformer = DataTransfomer(func="ohe-min-max")
        model_interface.get_num_output_nodes.return_value = np.zeros((1, 2))

        explainer = DicePyTorch(data_interface, model_interface)

        assert explainer.data_interface.ohe_encoded_feature_names == [
            "age",
            "hours_per_week",
            "workclass_Government",
            "workclass_Private",
            "workclass_Self-Employed",
        ]
        assert explainer.num_output_nodes == 2
