"""Module containing all required information about the interface between raw (or transformed)
public data and DiCE explainers."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException
from sklearn.preprocessing import LabelEncoder

from dice_ml.data_interfaces.base_data_interface import _BaseData
from dice_ml.utils.exception import SystemException


class PublicData(_BaseData):
    """A data interface for public data. This class is an interface to DiCE explainers
       and contains methods to transform user-fed raw data into the format a DiCE explainer
       requires, and vice versa."""

    def __init__(self, params):
        """Init method

        :param dataframe: The train dataframe used by explainer method.
        :param continuous_features: List of names of continuous features. The remaining features are categorical features.
        :param outcome_name: Outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range in list as values.
                                           Defaults to the range inferred from training data.
        :param continuous_features_precision (optional): Dictionary with feature names as keys and precisions as values.
        :param data_name (optional): Dataset name
        """
        self._validate_and_set_outcome_name(params=params)
        self._validate_and_set_dataframe(params=params)
        self._validate_and_set_continuous_features(params=params)

        self.feature_names = [
            name for name in self.data_df.columns.tolist() if name != self.outcome_name]

        self.number_of_features = len(self.feature_names)

        if len(set(self.continuous_feature_names) - set(self.feature_names)) != 0:
            raise UserConfigValidationException(
                "continuous_features contains some feature names which are not part of columns in dataframe"
            )

        self.categorical_feature_names = [name for name in self.data_df.columns.tolist(
        ) if name not in self.continuous_feature_names + [self.outcome_name]]

        self.categorical_feature_indexes = [self.data_df.columns.get_loc(
            name) for name in self.categorical_feature_names if name in self.data_df]

        self._validate_and_set_continuous_features_precision(params=params)
        self.data_df = self._set_feature_dtypes(
                self.data_df,
                self.categorical_feature_names,
                self.continuous_feature_names)

        self._validate_and_set_permitted_range(params=params)
        self._validate_and_set_data_name(params=params)

    def _validate_and_set_dataframe(self, params):
        """Validate and set the dataframe."""
        if 'dataframe' not in params:
            raise ValueError("dataframe not found in params")

        if isinstance(params['dataframe'], pd.DataFrame):
            self.data_df = params['dataframe'].copy()
        else:
            raise ValueError("should provide a pandas dataframe")

        if 'outcome_name' in params and params['outcome_name'] not in self.data_df.columns.tolist():
            raise UserConfigValidationException(
                "outcome_name {0} not found in {1}".format(
                    params['outcome_name'], ','.join(self.data_df.columns.tolist())
                )
            )

    def _validate_and_set_continuous_features(self, params):
        """Validate and set the list of continuous features."""
        if 'continuous_features' not in params:
            raise ValueError('continuous_features should be provided')

        if isinstance(params['continuous_features'], list):
            self.continuous_feature_names = params['continuous_features']
        else:
            raise ValueError(
                "should provide the name(s) of continuous features in the data as a list")

    def _validate_and_set_continuous_features_precision(self, params):
        """Validate and set the dictionary of precision for continuous features."""
        if 'continuous_features_precision' in params:
            self.continuous_features_precision = params['continuous_features_precision']

            if not hasattr(self, 'feature_names'):
                raise SystemException('Feature names not correctly set in public data interface')

            for continuous_features_precision_feature_name in self.continuous_features_precision:
                if continuous_features_precision_feature_name not in self.feature_names:
                    raise UserConfigValidationException(
                        "continuous_features_precision contains some feature names which are not part of columns in dataframe"
                    )
        else:
            self.continuous_features_precision = None

    def _set_feature_dtypes(self, data_df, categorical_feature_names,
                            continuous_feature_names):
        """Set the correct type of each feature column."""
        if len(categorical_feature_names) > 0:
            for feature in categorical_feature_names:
                data_df[feature] = data_df[feature].apply(str)
            data_df[categorical_feature_names] = data_df[categorical_feature_names].astype(
                'category')

        if len(continuous_feature_names) > 0:
            for feature in continuous_feature_names:
                if self.get_data_type(feature) == 'float':
                    data_df[feature] = data_df[feature].astype(
                        np.float32)
                else:
                    data_df[feature] = data_df[feature].astype(
                        np.int32)
        return data_df

    def get_features_range(self, permitted_range_input=None, features_dict=None):
        ranges = {}
        # Getting default ranges based on the dataset
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                self.data_df[feature_name].min(), self.data_df[feature_name].max()]
        for feature_name in self.categorical_feature_names:
            ranges[feature_name] = self.data_df[feature_name].unique().tolist()
        feature_ranges_orig = ranges.copy()
        # Overwriting the ranges for a feature if input provided
        if permitted_range_input is not None:
            for feature_name, feature_range in permitted_range_input.items():
                ranges[feature_name] = feature_range
        return ranges, feature_ranges_orig

    def get_data_type(self, col):
        """Infers data type of a continuous feature from the training data."""
        if (self.data_df[col].dtype == np.int64) or (self.data_df[col].dtype == np.int32) or \
                (self.data_df[col].dtype == np.int16) or (self.data_df[col].dtype == np.int8):
            return 'int'
        elif (self.data_df[col].dtype == np.float64) or (self.data_df[col].dtype == np.float32) or \
                (self.data_df[col].dtype == np.float16):
            return 'float'
        else:
            raise ValueError("Unknown data type of feature %s: must be int or float" % col)

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        if isinstance(df, pd.DataFrame) or isinstance(df, dict):
            for feature_name in self.continuous_feature_names:
                max_value = self.data_df[feature_name].max()
                min_value = self.data_df[feature_name].min()
                if min_value == max_value:
                    result[feature_name] = 0
                else:
                    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result = result.astype('float')
            for feature_index in self.continuous_feature_indexes:
                feature_name = self.feature_names[feature_index]
                max_value = self.data_df[feature_name].max()
                min_value = self.data_df[feature_name].min()
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
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (
                                           df[feature_name] * (max_value - min_value)) + min_value
        return result

    def get_valid_feature_range(self, feature_range_input, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized
        form. Assumes that all features are already encoded to numerical form
        such that the number of features remains the same.

        # TODO needs work adhere to label encoded max and to support permitted_range for
        both continuous and discrete when provided in _generate_counterfactuals.
        """
        feature_range = {}

        for _, feature_name in enumerate(self.feature_names):
            feature_range[feature_name] = []
            if feature_name in self.continuous_feature_names:
                max_value = self.data_df[feature_name].max()
                min_value = self.data_df[feature_name].min()

                if normalized:
                    minx = (feature_range_input[feature_name]
                            [0] - min_value) / (max_value - min_value)
                    maxx = (feature_range_input[feature_name]
                            [1] - min_value) / (max_value - min_value)
                else:
                    minx = feature_range_input[feature_name][0]
                    maxx = feature_range_input[feature_name][1]
                feature_range[feature_name].append(minx)
                feature_range[feature_name].append(maxx)
            else:
                # categorical features
                feature_range[feature_name] = feature_range_input[feature_name]
        return feature_range

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0] * len(self.ohe_encoded_feature_names)])
        maxx = np.array([[1.0] * len(self.ohe_encoded_feature_names)])

        for idx, feature_name in enumerate(self.continuous_feature_names):
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()

            if normalized:
                minx[0][idx] = (self.permitted_range[feature_name]
                                [0] - min_value) / (max_value - min_value)
                maxx[0][idx] = (self.permitted_range[feature_name]
                                [1] - min_value) / (max_value - min_value)
            else:
                minx[0][idx] = self.permitted_range[feature_name][0]
                maxx[0][idx] = self.permitted_range[feature_name][1]
        return minx, maxx

    def get_mads(self, normalized=False):
        """Computes Median Absolute Deviation of features."""
        mads = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(self.data_df[feature].values - np.median(self.data_df[feature].values)))
        else:
            normalized_train_df = self.normalize_data(self.data_df)
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(normalized_train_df[feature].values - np.median(normalized_train_df[feature].values)))
        return mads

    def get_valid_mads(self, normalized=False, display_warnings=False, return_mads=True):
        """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
        mads = self.get_mads(normalized=normalized)
        for feature in mads:
            if mads[feature] <= 0:
                mads[feature] = 1.0
                if display_warnings:
                    logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)

        if return_mads:
            return mads

    def get_quantiles_from_training_data(self, quantile=0.05, normalized=False):
        """Computes required quantile of Absolute Deviations of features."""

        quantiles = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                quantiles[feature] = np.quantile(
                    abs(list(set(self.data_df[feature].tolist())) - np.median(
                        list(set(self.data_df[feature].tolist())))), quantile)
        else:
            normalized_train_df = self.normalize_data(self.data_df)
            for feature in self.continuous_feature_names:
                quantiles[feature] = np.quantile(
                    abs(list(set(normalized_train_df[feature].tolist())) - np.median(
                        list(set(normalized_train_df[feature].tolist())))), quantile)
        return quantiles

    def create_ohe_params(self, one_hot_encoded_data):
        if len(self.categorical_feature_names) > 0:
            self.ohe_encoded_feature_names = [x for x in one_hot_encoded_data.columns.tolist(
                ) if x not in np.array([self.outcome_name])]
        else:
            # one-hot-encoded data is same as original data if there is no categorical features.
            self.ohe_encoded_feature_names = [feat for feat in self.feature_names]

    def get_data_params_for_gradient_dice(self):
        """Gets all data related params for DiCE."""

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

        return minx, maxx, encoded_categorical_feature_indexes, encoded_continuous_feature_indexes, cont_minx, \
            cont_maxx, cont_precisions

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
        # TODO: add encoding as a parameter and use the function get_indexes_of_features_to_vary for label encoding too
        if features_to_vary == "all":
            return [i for i in range(len(self.ohe_encoded_feature_names))]
        else:
            ixs = []
            encoded_cats_ixs = self.get_encoded_categorical_feature_indexes()
            encoded_cats_ixs = [item for sublist in encoded_cats_ixs for item in sublist]
            for colidx, col in enumerate(self.ohe_encoded_feature_names):
                if colidx in encoded_cats_ixs and col.startswith(tuple(features_to_vary)):
                    ixs.append(colidx)
                elif colidx not in encoded_cats_ixs and col in features_to_vary:
                    ixs.append(colidx)
            return ixs

    def fit_label_encoders(self):
        labelencoders = {}
        for column in self.categorical_feature_names:
            labelencoders[column] = LabelEncoder()
            labelencoders[column] = labelencoders[column].fit(self.data_df[column])
        return labelencoders

    def from_label(self, data):
        """Transforms label encoded data back to categorical values"""
        out = data.copy()
        if isinstance(data, pd.DataFrame) or isinstance(data, dict):
            for column in self.categorical_feature_names:
                out[column] = self.labelencoder[column].inverse_transform(out[column].round().astype(int).tolist())
            return out
        elif isinstance(data, list):
            for c in self.categorical_feature_indexes:
                out[c] = self.labelencoder[self.feature_names[c]].inverse_transform([round(out[c])])[0]
            return out

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

    def get_decimal_precisions(self, output_type="list"):
        """"Gets the precision of continuous features in the data."""
        # if the precision of a continuous feature is not given, we use the maximum precision of the modes to capture the
        # precision of majority of values in the column.
        precisions_dict = defaultdict(int)
        precisions = [0] * len(self.feature_names)
        for ix, col in enumerate(self.continuous_feature_names):
            if (self.continuous_features_precision is not None) and (col in self.continuous_features_precision):
                precisions[ix] = self.continuous_features_precision[col]
                precisions_dict[col] = self.continuous_features_precision[col]
            elif self.data_df[col].dtype == np.float32 or self.data_df[col].dtype == np.float64:
                modes = self.data_df[col].mode()
                maxp = len(str(modes[0]).split('.')[1])  # maxp stores the maximum precision of the modes
                for mx in range(len(modes)):
                    prec = len(str(modes[mx]).split('.')[1])
                    if prec > maxp:
                        maxp = prec
                precisions[ix] = maxp
                precisions_dict[col] = maxp
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
            levels.append(self.data_df[cat_feature].cat.categories.tolist())

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

    def prepare_query_instance(self, query_instance):
        """Prepares user defined test input(s) for DiCE."""
        test = self.query_instance_to_df(query_instance)
        test = test.reset_index(drop=True)
        # encode categorical and numerical columns
        test = self._set_feature_dtypes(test,
                                        self.categorical_feature_names,
                                        self.continuous_feature_names)
        return test

    def get_ohe_min_max_normalized_data(self, query_instance):
        """Transforms query_instance into one-hot-encoded and min-max normalized data. query_instance should be a dict,
           a dataframe, a list, or a list of dicts"""
        query_instance = self.prepare_query_instance(query_instance)
        ohe_base_df = self.prepare_df_for_ohe_encoding()
        temp = pd.concat([ohe_base_df, query_instance], ignore_index=True, sort=False)
        temp = self.one_hot_encode_data(temp)
        temp = temp.tail(query_instance.shape[0]).reset_index(drop=True)
        # returns a pandas dataframe with all numeric values
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
        return pd.get_dummies(self.data_df[self.feature_names]).columns
