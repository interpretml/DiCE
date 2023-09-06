import pandas as pd

from dice_ml.utils.helpers import load_adult_income_dataset


class TestHelpers:
    def test_load_adult_income_dataset(self):
        adult_data = load_adult_income_dataset()
        assert adult_data is not None
        assert isinstance(adult_data, pd.DataFrame)
