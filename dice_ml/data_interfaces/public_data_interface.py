"""Module containing all required information about the raw or transformed public data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class PublicData:
    """A data interface for public data."""

    def __init__(self, params):
        """Init method

        :param dataframe: The train dataframe used by explainer method.
        :param continuous_features: List of names of continuous features. The remaining features are categorical features.
        :param outcome_name: Outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data.
        :param continuous_features_precision (optional): Dictionary with feature names as keys and precisions as values.
        :param data_name (optional): Dataset name

        """

        if isinstance(params['dataframe'], pd.DataFrame):
            self.data_df = params['dataframe']
        else:
            raise ValueError("should provide a pandas dataframe")

        if type(params['continuous_features']) is list:
            self.continuous_feature_names = params['continuous_features']
        else:
            raise ValueError(
                "should provide the name(s) of continuous features in the data")

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature")

        self.categorical_feature_names = [name for name in self.data_df.columns.tolist(
        ) if name not in self.continuous_feature_names + [self.outcome_name]]

        self.feature_names = [
            name for name in self.data_df.columns.tolist() if name != self.outcome_name]

        self.continuous_feature_indexes = [self.data_df.columns.get_loc(
            name) for name in self.continuous_feature_names if name in self.data_df]

        self.categorical_feature_indexes = [self.data_df.columns.get_loc(
            name) for name in self.categorical_feature_names if name in self.data_df]

        if 'continuous_features_precision' in params:
            self.continuous_features_precision = params['continuous_features_precision']
        else:
            self.continuous_features_precision = None

        if len(self.categorical_feature_names) > 0:
            for feature in self.categorical_feature_names:
                self.data_df[feature] = self.data_df[feature].apply(str)
            self.data_df[self.categorical_feature_names] = self.data_df[self.categorical_feature_names].astype(
                'category')

        if len(self.continuous_feature_names) > 0:
            for feature in self.continuous_feature_names:
                if self.get_data_type(feature) == 'float':
                    self.data_df[feature] = self.data_df[feature].astype(
                        np.float32)
                else:
                    self.data_df[feature] = self.data_df[feature].astype(
                        np.int32)

        # should move the below snippet to gradient based dice interfaces
                                # self.one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
                                # self.encoded_feature_names = [x for x in self.one_hot_encoded_data.columns.tolist(
                                #     ) if x not in np.array([self.outcome_name])]

        # should move the below snippet to model agnostic dice interfaces
                                # # Initializing a label encoder to obtain label-encoded values for categorical variables
                                # self.labelencoder = {}
                                #
                                # self.label_encoded_data = self.data_df.copy()
                                #
                                # for column in self.categorical_feature_names:
                                #     self.labelencoder[column] = LabelEncoder()
                                #     self.label_encoded_data[column] = self.labelencoder[column].fit_transform(self.data_df[column])

        self.permitted_range = self.get_features_range()
        if 'permitted_range' in params:
            for feature_name, feature_range in params['permitted_range'].items():
                self.permitted_range[feature_name] = feature_range
            if not self.check_features_range():
                raise ValueError(
                    "permitted range of features should be within their original range")

        # should move the below snippet to model agnostic dice interfaces
                                # self.max_range = -np.inf
                                # for feature in self.continuous_feature_names:
                                #     self.max_range = max(self.max_range, self.permitted_range[feature][1])

        if 'data_name' in params:
            self.data_name = params['data_name']
        else:
            self.data_name = 'mydata'

    def check_features_range(self):
        for feature in self.continuous_feature_names:
            if feature in self.permitted_range:
                min_value = self.data_df[feature].min()
                max_value = self.data_df[feature].max()

                if self.permitted_range[feature][0] < min_value or self.permitted_range[feature][1] > max_value:
                    return False
            else:
                self.permitted_range[feature] = [self.data_df[feature].min(), self.data_df[feature].max()]
        return True

    def get_features_range(self):
        ranges = {}
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                self.data_df[feature_name].min(), self.data_df[feature_name].max()]
        return ranges

    def get_data_type(self, col):
        """Infers data type of a continuous feature from the training data."""
        if ((self.data_df[col].dtype == np.int64) or (self.data_df[col].dtype == np.int32)):
            return 'int'
        elif ((self.data_df[col].dtype == np.float64) or (self.data_df[col].dtype == np.float32)):
            return 'float'
        else:
            raise ValueError("Unknown data type of feature %s: must be int or float" % col)

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (
                                           df[feature_name] - min_value) / (max_value - min_value)
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

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0] * len(self.encoded_feature_names)])
        maxx = np.array([[1.0] * len(self.encoded_feature_names)])

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

    def get_data_params_for_gradient_dice(self):
        """Gets all data related params for DiCE."""

        if len(self.categorical_feature_names) > 0:
            one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
            self.encoded_feature_names = [x for x in one_hot_encoded_data.columns.tolist(
                ) if x not in np.array([self.outcome_name])]
        else:
            # one-hot-encoded data is same as original data if there is no categorical features.
            self.encoded_feature_names = [feat for feat in self.feature_names]

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
        cont_precisions = [self.get_decimal_precisions()[ix] for ix in encoded_continuous_feature_indexes]

        return minx, maxx, encoded_categorical_feature_indexes, encoded_continuous_feature_indexes, cont_minx, cont_maxx, cont_precisions

    def get_encoded_categorical_feature_indexes(self):
        """Gets the column indexes categorical features after one-hot-encoding."""
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.encoded_feature_names.index(
                col) for col in self.encoded_feature_names if col.startswith(col_parent) and
                                                              col not in self.continuous_feature_names]
            cols.append(temp)
        return cols

    def get_indexes_of_features_to_vary(self, features_to_vary='all'):
        """Gets indexes from feature names of one-hot-encoded data."""
        # TODO: add encoding as a parameter and use the function get_indexes_of_features_to_vary for label encoding too
        if features_to_vary == "all":
            return [i for i in range(len(self.encoded_feature_names))]
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

            # then, recreate original data by removing the suffixes - based on the GitHub issue comment: https://github.com/pandas-dev/pandas/issues/8745#issuecomment-417861271
            cols, labs = [[c.replace(
                x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
            out[feat] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decimal_precisions(self):
        """"Gets the precision of continuous features in the data."""
        # if the precision of a continuous feature is not given, we use the maximum precision of the modes to capture the precision of majority of values in the column.
        precisions = [0] * len(self.feature_names)
        for ix, col in enumerate(self.continuous_feature_names):
            if ((self.continuous_features_precision is not None) and (col in self.continuous_features_precision)):
                precisions[ix] = self.continuous_features_precision[col]
            elif ((self.data_df[col].dtype == np.float32) or (self.data_df[col].dtype == np.float64)):
                modes = self.data_df[col].mode()
                maxp = len(str(modes[0]).split('.')[1])  # maxp stores the maximum precision of the modes
                for mx in range(len(modes)):
                    prec = len(str(modes[mx]).split('.')[1])
                    if prec > maxp:
                        maxp = prec
                precisions[ix] = maxp
        return precisions

    def get_decoded_data(self, data, encoding='one-hot'):
        """Gets the original data from encoded data."""
        if len(data) == 0:
            return data
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            if encoding == 'one-hot':
                data = pd.DataFrame(data=data, index=index,
                                    columns=self.encoded_feature_names)
                return self.from_dummies(data)

            elif encoding == 'label':
                data = pd.DataFrame(data=data, index=index,
                                    columns=self.feature_names)
                return data

    def prepare_df_for_encoding(self):
        """Facilitates prepare_query_instance() function."""
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(self.data_df[cat_feature].cat.categories.tolist())

        if len(colnames) > 0:
            df = pd.DataFrame({colnames[0]: levels[0]})
        else:
            df = pd.DataFrame()

        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = self.continuous_feature_names
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        return df

    def prepare_query_instance(self, query_instance, encoding='one-hot'):
        """Prepares user defined test input(s) for DiCE."""
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

        test = test.reset_index(drop=True)

        if encoding == 'label':
            for column in self.categorical_feature_names:
                test[column] = self.labelencoder[column].transform(test[column])
            return self.normalize_data(test)

        elif encoding == 'one-hot':
            temp = self.prepare_df_for_encoding()
            temp = temp.append(test, ignore_index=True, sort=False)
            temp = self.one_hot_encode_data(temp)
            temp = self.normalize_data(temp)

            return temp.tail(test.shape[0]).reset_index(drop=True)
