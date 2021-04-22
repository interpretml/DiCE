class DummyDataInterface:
    def __init__(self, outcome_name, data_df=None):
        self.outcome_name = outcome_name
        if data_df is not None:
            self.data_df = data_df

    def to_json(self):
        return {
            'outcome_name': self.outcome_name,
            'data_df': self.data_df
        }
