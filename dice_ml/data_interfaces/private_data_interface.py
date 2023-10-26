"""Module containing meta data information about private data."""

import collections
import logging
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dice_ml.data_interfaces.base_data_interface import _BaseData


class PrivateData(_BaseData):
    """A data interface for private data with meta information."""

    def __init__(self, params):
        """Init method

        :param features: Dictionary or OrderedDict with feature names as keys and range in int/float
                         (for continuous features) or categories in string (for categorical features)
                         as values. For python version <=3.6, should provide only an OrderedDict.
        :param outcome_name: Outcome feature name.
        :param type_and_precision (optional): Dictionary with continuous feature names as keys.
                                              If the feature is of type int, just string 'int' should be provided,
                                              if the feature is of type float, a list of type and precision should be
                                              provided. For instance, type_and_precision: {cont_f1: 'int',
                                              cont_f2: ['float', 2]} for continuous features cont_f1 and cont_f2 of
                                              type int and float (and precision up to 2 decimal places) respectively.
                                              Default value is None and all features are treated as int.
        :param mad (optional): Dictionary with feature names as keys and corresponding Median Absolute Deviations (MAD)
                               as values.
                               Default MAD value is 1 for all features.
        :param data_name (optional): Dataset name
        """
        if sys.version_info > (3, 6, 0) and isinstance(params['features'], (dict, collections.OrderedDict)):
            features_dict = params['features']
        elif sys.version_info <= (3, 6, 0) and isinstance(params['features'], collections.OrderedDict):
            features_dict = params['features']
        else:
            raise ValueError(
                "should provide dictionary with feature names as keys and range"
                "(for continuous features) or categories (for categorical features) as values. "
                "For python version <3.6, should provide an OrderedDict")

        self._validate_and_set_outcome_name(params=params)
        self._validate_and_set_type_and_precision(params=params)

        self.continuous_feature_names = []
        self.categorical_feature_names = []
        self.categorical_levels = {}

        for feature in features_dict:
            if isinstance(features_dict[feature][0], (int, float)):  # continuous feature
                self.continuous_feature_names.append(feature)
            else:
                self.categorical_feature_names.append(feature)
                self.categorical_levels[feature] = features_dict[feature]

        self._validate_and_set_mad(params=params)
        self._validate_and_set_permitted_range(params=params, features_dict=features_dict)
        self.feature_names = list(features_dict.keys())
        self.number_of_features = len(self.feature_names)

        self.continuous_feature_indexes = [list(features_dict.keys()).index(
            name) for name in self.continuous_feature_names if name in features_dict]

        self.categorical_feature_indexes = [list(features_dict.keys()).index(
            name) for name in self.categorical_feature_names if name in features_dict]

        for feature_name in self.continuous_feature_names:
            if feature_name not in self.type_and_precision:
                self.type_and_precision[feature_name] = 'int'

        self._validate_and_set_data_name(params=params)

    def _validate_and_set_type_and_precision(self, params):
        """Validate and set the type and precision."""
        if 'type_and_precision' in params:
            self.type_and_precision = params['type_and_precision']
        else:
            self.type_and_precision = {}

    def _validate_and_set_mad(self, params):
        """Validate and set the MAD."""
        if 'mad' in params:
            self.mad = params['mad']
        else:
            self.mad = {}

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        if isinstance(df, pd.DataFrame) or isinstance(df, dict):
            for feature_name in self.continuous_feature_names:
                max_value = self.permitted_range[feature_name][1]
                min_value = self.permitted_range[feature_name][0]
                if min_value == max_value:
                    result[feature_name] = 0
                else:
                    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result = result.astype('float')
            for feature_index in self.continuous_feature_indexes:
                feature_name = self.feature_names[feature_index]
                max_value = self.permitted_range[feature_name][1]
                min_value = self.permitted_range[feature_name][0]
                if len(df.shape) == 1:
                    if min_value == max_value:
                        value = 0
                    else:
                        value = (df[feature_index] - min_value) / (max_value - min_value)
                    result[feature_index] = value
                else:
                    if min_value == max_value:
                        result[:, feature_index] = np.zeros(len(df[:, feature_index]))
                    else:
                        result[:, feature_index] = (df[:, feature_index] - min_value) / (max_value - min_value)
        return result

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        if len(df) == 0:
            return df
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.permitted_range[feature_name][1]
            min_value = self.permitted_range[feature_name][0]
            result[feature_name] = (
                df[feature_name]*(max_value - min_value)) + min_value
        return result

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0]*len(self.ohe_encoded_feature_names)])
        maxx = np.array([[1.0]*len(self.ohe_encoded_feature_names)])

        if normalized:
            return minx, maxx
        else:
            for idx, feature_name in enumerate(self.continuous_feature_names):
                minx[0][idx] = self.permitted_range[feature_name][0]
                maxx[0][idx] = self.permitted_range[feature_name][1]
            return minx, maxx

    def get_mads(self, normalized=True):
        """Computes Median Absolute Deviation of features."""
        if normalized is False:
            return self.mad.copy()
        else:
            mads = {}
            for feature in self.continuous_feature_names:
                if feature in self.mad:
                    mads[feature] = (self.mad[feature])/(self.permitted_range[feature][1] - self.permitted_range[feature][0])
            return mads

    def get_valid_mads(self, normalized=False, display_warnings=False, return_mads=True):
        """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
        mads = self.get_mads(normalized=normalized)
        for feature in self.continuous_feature_names:
            if feature in mads:
                if mads[feature] <= 0:
                    mads[feature] = 1.0
                    if display_warnings:
                        logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)
            else:
                mads[feature] = 1.0
                if display_warnings:
                    logging.info(" MAD is not given for feature %s, so using 1.0 as MAD instead.", feature)

        if return_mads:
            return mads

    def get_features_range(self, permitted_range_input=None, features_dict=None):
        if features_dict is None:
            features_dict = self.permitted_range

        ranges = {}
        # Getting default ranges based on the dataset
        for feature in features_dict:
            if type(features_dict[feature][0]) is int:  # continuous feature
                ranges[feature] = features_dict[feature]
            else:
                ranges[feature] = features_dict[feature]
        feature_ranges_orig = ranges.copy()
        # Overwriting the ranges for a feature if input provided
        if permitted_range_input is not None:
            for feature_name, feature_range in permitted_range_input.items():
                ranges[feature_name] = feature_range
        return ranges, feature_ranges_orig

    def create_ohe_params(self, one_hot_encoded_data=None):
        if len(self.categorical_feature_names) > 0:
            # simulating sklearn's one-hot-encoding
            # continuous features on the left
            self.ohe_encoded_feature_names = [
                feature for feature in self.continuous_feature_names]
            for feature_name in self.categorical_feature_names:
                for category in sorted(self.categorical_levels[feature_name]):
                    self.ohe_encoded_feature_names.append(
                        feature_name+'_'+category)
        else:
            # one-hot-encoded data is same as original data if there is no categorical features.
            self.ohe_encoded_feature_names = [feat for feat in self.feature_names]

        # base dataframe for doing one-hot-encoding
        # ohe_encoded_feature_names and ohe_base_df are created (and stored as data class's parameters)
        # when get_data_params_for_gradient_dice() is called from gradient-based DiCE explainers
        self.ohe_base_df = self.prepare_df_for_ohe_encoding()

    def get_data_params_for_gradient_dice(self):
        """Gets all data related params for DiCE."""

        self.create_ohe_params()
        minx, maxx = self.get_minx_maxx(normalized=True)

        # get the column indexes of categorical and continuous features after one-hot-encoding
        encoded_categorical_feature_indexes = self.get_encoded_categorical_feature_indexes()
        flattened_indexes = [item for sublist in encoded_categorical_feature_indexes for item in sublist]
        encoded_continuous_feature_indexes = [ix for ix in range(len(minx[0])) if ix not in flattened_indexes]

        # min and max for continuous features in original scale
        org_minx, org_maxx = self.get_minx_maxx(normalized=False)
        cont_minx = list(org_minx[0][encoded_continuous_feature_indexes])
        cont_maxx = list(org_maxx[0][encoded_continuous_feature_indexes])

        # decimal precisions for continuous features
        cont_precisions = [self.get_decimal_precisions()[ix] for ix in range(len(self.continuous_feature_names))]

        return minx, maxx, encoded_categorical_feature_indexes, encoded_continuous_feature_indexes, \
            cont_minx, cont_maxx, cont_precisions

    def get_encoded_categorical_feature_indexes(self):
        """Gets the column indexes categorical features after one-hot-encoding."""
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.ohe_encoded_feature_names.index(
                col) for col in self.ohe_encoded_feature_names if col.startswith(col_parent) and
                   col not in self.continuous_feature_names]
            cols.append(temp)
        return cols

    def get_indexes_of_features_to_vary(self, features_to_vary='all'):
        """Gets indexes from feature names of one-hot-encoded data."""
        if features_to_vary == "all":
            return [i for i in range(len(self.ohe_encoded_feature_names))]
        else:
            ixs = []
            encoded_cats_ixs = self.get_encoded_categorical_feature_indexes()
            encoded_cats_ixs = [item for sublist in encoded_cats_ixs for item in sublist]
            for colidx, col in enumerate(self.encoded_feature_names):
                if colidx in encoded_cats_ixs and col.startswith(tuple(features_to_vary)):
                    ixs.append(colidx)
                elif colidx not in encoded_cats_ixs and col in features_to_vary:
                    ixs.append(colidx)
            return ixs

    def fit_label_encoders(self):
        labelencoders = {}
        for column in self.categorical_feature_names:
            labelencoders[column] = LabelEncoder()
            labelencoders[column].fit(self.permitted_range[column])
        return labelencoders

    def from_label(self, data):
        """Transforms label encoded data back to categorical values"""
        out = data.copy()
        if isinstance(data, pd.DataFrame) or isinstance(data, dict):
            for column in self.categorical_feature_names:
                out[column] = self.labelencoder[column].inverse_transform(out[column].round().astype(int).tolist())
            return out
        elif isinstance(data, list):
            for column in self.categorical_feature_indexes:
                out[column] = self.labelencoder[self.feature_names[column]].inverse_transform([round(out[column])])[0]
            return out

    def from_dummies(self, data, prefix_sep='_'):
        """Gets the original data from dummy encoded data with k levels."""
        out = data.copy()
        for feature_name in self.categorical_feature_names:
            cols, labs = [[c.replace(
                x, "") for c in data.columns if feature_name+prefix_sep in c] for x in ["", feature_name+prefix_sep]]
            out[feature_name] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decimal_precisions(self, output_type="list"):
        """"Gets the precision of continuous features in the data."""
        precisions_dict = defaultdict(int)
        precisions = [0]*len(self.feature_names)
        for ix, feature_name in enumerate(self.continuous_feature_names):
            type_prec = self.type_and_precision[feature_name]
            if type_prec == 'int':
                prec = 0
            else:
                prec = self.type_and_precision[feature_name][1]
            precisions[ix] = prec
            precisions_dict[feature_name] = prec
        if output_type == "list":
            return precisions
        elif output_type == "dict":
            return precisions_dict

    def get_decoded_data(self, data, encoding='one-hot'):
        """Gets the original data from encoded data."""
        if len(data) == 0:
            return data

        index = [i for i in range(0, len(data))]
        if encoding == 'one-hot':
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, np.ndarray):
                data = pd.DataFrame(data=data, index=index,
                                    columns=self.ohe_encoded_feature_names)
                return data
            else:
                raise ValueError("data should be a pandas dataframe or a numpy array")

        elif encoding == 'label':
            data = pd.DataFrame(data=data, index=index,
                                columns=self.feature_names)
            return data

    def prepare_df_for_ohe_encoding(self):
        """Create base dataframe to do OHE for a single instance or a set of instances"""
        levels = []
        colnames = [feat for feat in self.categorical_feature_names]
        for cat_feature in colnames:
            levels.append(self.categorical_levels[cat_feature])

        if len(colnames) > 0:
            df = pd.DataFrame({colnames[0]: levels[0]})
        else:
            df = pd.DataFrame()

        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = [feat for feat in self.continuous_feature_names]
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        return df

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

    def prepare_query_instance(self, query_instance):
        """Prepares user defined test input(s) for DiCE."""
        test = self.query_instance_to_df(query_instance)
        test = test.reset_index(drop=True)
        return test

    def get_ohe_min_max_normalized_data(self, query_instance):
        """Transforms query_instance into one-hot-encoded and min-max normalized data. query_instance should be a dict,
           a dataframe, a list, or a list of dicts"""
        query_instance = self.prepare_query_instance(query_instance)
        ohe_base_df = self.prepare_df_for_ohe_encoding()
        temp = pd.concat([ohe_base_df, query_instance], ignore_index=True, sort=False)
        temp = self.one_hot_encode_data(temp)
        temp = temp.tail(query_instance.shape[0]).reset_index(drop=True)
        # returns a pandas dataframe
        return self.normalize_data(temp).apply(pd.to_numeric)

    def get_inverse_ohe_min_max_normalized_data(self, transformed_data):
        """Transforms one-hot-encoded and min-max normalized data into raw user-fed data format. transformed_data
           should be a dataframe or an array"""
        raw_data = self.from_dummies(transformed_data)
        raw_data = self.de_normalize_data(raw_data)
        precisions = self.get_decimal_precisions()
        for ix, feature in enumerate(self.continuous_feature_names):
            raw_data[feature] = raw_data[feature].astype(float).round(precisions[ix])
        raw_data = raw_data[self.feature_names]
        # returns a pandas dataframe
        return raw_data

    def get_all_dummy_colnames(self):
        max_vals = max([len(val) for col, val in self.permitted_range.items()])
        sample_data_dict = {col: np.resize(val, max_vals) for col, val in self.permitted_range.items()}
        sample_df = pd.DataFrame(sample_data_dict)
        return pd.get_dummies(sample_df).columns

    def get_valid_feature_range(self, feature_range_input, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized
        form. Assumes that all features are already encoded to numerical form
        such that the number of features remains the same.

        # TODO needs work adhere to label encoded max and to support permitted_range for
        both continuous and discrete when provided in _generate_counterfactuals.
        """
        if normalized:
            raise NotImplementedError("Normalized feature range not supported for private data interface")
        return feature_range_input
