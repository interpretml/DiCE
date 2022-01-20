"""Module containing base class for data interfaces for dice-ml."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class _BaseData(ABC):

    def _validate_and_set_data_name(self, params):
        """Validate and set the data name."""
        if 'data_name' in params:
            self.data_name = params['data_name']
        else:
            self.data_name = 'mydata'

    def _validate_and_set_outcome_name(self, params):
        """Validate and set the outcome name."""
        if 'outcome_name' not in params:
            raise ValueError("should provide the name of outcome feature")

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature as a string")

    def set_continuous_feature_indexes(self, query_instance):
        """Remaps continuous feature indices based on the query instance"""
        self.continuous_feature_indexes = [query_instance.columns.get_loc(name) for name in
                                           self.continuous_feature_names]

    def from_dummies(self, data, prefix_sep='_'):
        """Gets the original data from dummy encoded data with k levels."""
        out = data.copy()
        for feat in self.categorical_feature_names:
            # first, derive column names in the one-hot-encoded data from the original data
            cat_col_values = []
            for val in list(self.data_df[feat].unique()):
                cat_col_values.append(feat + prefix_sep + str(
                    val))  # join original feature name and its unique values , ex: education_school
            match_cols = [c for c in data.columns if
                          c in cat_col_values]  # check for the above matching columns in the encoded data

            # then, recreate original data by removing the suffixes - based on the GitHub issue comment:
            # https://github.com/pandas-dev/pandas/issues/8745#issuecomment-417861271
            cols, labs = [[c.replace(
                x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
            out[feat] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

    def get_decoded_data(self, data, encoding='one-hot'):
        """Gets the original data from encoded data."""
        if len(data) == 0:
            return data

        index = [i for i in range(0, len(data))]
        if encoding == 'one-hot':
            if isinstance(data, pd.DataFrame):
                return self.from_dummies(data)
            elif isinstance(data, np.ndarray):
                data = pd.DataFrame(data=data, index=index,
                                    columns=self.ohe_encoded_feature_names)
                return self.from_dummies(data)
            else:
                raise ValueError("data should be a pandas dataframe or a numpy array")

        elif encoding == 'label':
            data = pd.DataFrame(data=data, index=index,
                                columns=self.feature_names)
            return data

    @abstractmethod
    def __init__(self, params):
        """The init method needs to be implemented by the inherting classes."""
        pass
