"""Module containing base class for data interfaces for dice-ml."""

from abc import ABC, abstractmethod

import pandas as pd
from raiutils.exceptions import UserConfigValidationException

from dice_ml.utils.exception import SystemException


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

        if isinstance(params['outcome_name'], str):
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature as a string")

    def set_continuous_feature_indexes(self, query_instance):
        """Remaps continuous feature indices based on the query instance"""
        self.continuous_feature_indexes = [query_instance.columns.get_loc(name) for name in
                                           self.continuous_feature_names]

    def check_features_to_vary(self, features_to_vary):
        if features_to_vary is not None and features_to_vary != 'all':
            not_training_features = set(features_to_vary) - set(self.feature_names)
            if len(not_training_features) > 0:
                raise UserConfigValidationException("Got features {0} which are not present in training data".format(
                    not_training_features))

    def check_permitted_range(self, permitted_range):
        if permitted_range is not None:
            permitted_range_features = list(permitted_range)
            not_training_features = set(permitted_range_features) - set(self.feature_names)
            if len(not_training_features) > 0:
                raise UserConfigValidationException("Got features {0} which are not present in training data".format(
                    not_training_features))

            for feature in permitted_range_features:
                if feature in self.categorical_feature_names:
                    train_categories = self.permitted_range[feature]
                    for test_category in permitted_range[feature]:
                        if test_category not in train_categories:
                            raise UserConfigValidationException(
                                'The category {0} does not occur in the training data for feature {1}.'
                                ' Allowed categories are {2}'.format(test_category, feature, train_categories))

    def _validate_and_set_permitted_range(self, params, features_dict=None):
        """Validate and set the dictionary of permitted ranges for continuous features."""
        input_permitted_range = None
        if 'permitted_range' in params:
            input_permitted_range = params['permitted_range']

            if not hasattr(self, 'feature_names'):
                raise SystemException('Feature names not correctly set in public data interface')

            for input_permitted_range_feature_name in input_permitted_range:
                if input_permitted_range_feature_name not in self.feature_names:
                    raise UserConfigValidationException(
                        "permitted_range contains some feature names which are not part of columns in dataframe"
                    )
        self.permitted_range, _ = self.get_features_range(input_permitted_range, features_dict)

    def ensure_consistent_type(self, output_df, query_instance):
        qdf = self.query_instance_to_df(query_instance)
        output_df = output_df.astype(qdf.dtypes.to_dict())
        return output_df

    def query_instance_to_df(self, query_instance):
        if isinstance(query_instance, list):
            if isinstance(query_instance[0], dict):  # prepare a list of query instances
                test = pd.DataFrame(query_instance, columns=self.feature_names)

            else:  # prepare a single query instance in list
                query_instance = {'row1': query_instance}
                test = pd.DataFrame.from_dict(
                    query_instance, orient='index', columns=self.feature_names)

        elif isinstance(query_instance, dict):
            test = pd.DataFrame({k: [v] for k, v in query_instance.items()}, columns=self.feature_names)

        elif isinstance(query_instance, pd.DataFrame):
            test = query_instance.copy()

        else:
            raise ValueError("Query instance should be a dict, a pandas dataframe, a list, or a list of dicts")
        return test

    @abstractmethod
    def __init__(self, params):
        """The init method needs to be implemented by the inherting classes."""
        pass
