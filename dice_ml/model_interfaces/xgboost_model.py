import xgboost as xgb

from dice_ml.constants import ModelTypes
from dice_ml.model_interfaces.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='', func=None, kw_args=None):
        super().__init__(model=model, model_path=model_path, backend='xgboost', func=func, kw_args=kw_args)
        if model is None and model_path:
            self.load_model()

    def load_model(self):
        if self.model_path != '':
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)

    def get_output(self, input_instance, model_score=True):
        input_instance = self.transformer.transform(input_instance)
        for col in input_instance.columns:
            input_instance[col] = input_instance[col].astype('int64')
        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.model.predict_proba(input_instance)
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

    def get_gradient(self):
        raise NotImplementedError("XGBoost does not support gradient calculation in this context")
