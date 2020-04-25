"""Module containing all required information about the raw or transformed public data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

import tensorflow as tf
from tensorflow import keras


class PublicData:
    """A data interface for public data."""

    def __init__(self, params):
        """Init method

        :param dataframe: Pandas DataFrame.
        :param continuous_features: List of names of continuous features. The remaining features are categorical features.
        :param outcome_name: Outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data.
        :param test_size (optional): Proportion of test set split. Defaults to 0.2.
        :param test_split_random_state (optional): Random state for train test split. Defaults to 17.
        :param continuous_features_precision (optional): Dictionary with feature names as keys and precisions as values.

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
        ) if name not in self.continuous_feature_names+[self.outcome_name]]

        self.feature_names = [
            name for name in self.data_df.columns.tolist() if name != self.outcome_name]

        self.continuous_feature_indexes = [self.data_df.columns.get_loc(
            name) for name in self.continuous_feature_names if name in self.data_df]

        self.categorical_feature_indexes = [self.data_df.columns.get_loc(
            name) for name in self.categorical_feature_names if name in self.data_df]

        if 'test_size' in params:
            self.test_size = params['test_size']
            if self.test_size > 1 or self.test_size < 0:
                raise ValueError(
                    "should provide a decimal between 0 and 1")
        else:
            self.test_size = 0.2

        if 'test_split_random_state' in params:
            self.test_split_random_state = params['test_split_random_state']
        else:
            self.test_split_random_state = 17

        if 'continuous_features_precision' in params:
            self.continuous_features_precision = params['continuous_features_precision']
        else:
            self.continuous_features_precision = None

        if len(self.categorical_feature_names) > 0:
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

        if len(self.categorical_feature_names) > 0:
            self.one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
            self.encoded_feature_names = [x for x in self.one_hot_encoded_data.columns.tolist(
            ) if x not in np.array([self.outcome_name])]
        else:
            # one-hot-encoded data is same as orignial data if there is no categorical features.
            self.one_hot_encoded_data = self.data_df
            self.encoded_feature_names = self.feature_names

        self.train_df, self.test_df = self.split_data(self.data_df)

        if 'permitted_range' in params:
            self.permitted_range = params['permitted_range']
            if not self.check_features_range():
                raise ValueError(
                    "permitted range of features should be within their original range")
        else:
            self.permitted_range = self.get_features_range()

    def check_features_range(self):
        for feature in self.continuous_feature_names:
            if feature in self.permitted_range:
                min_value = self.train_df[feature].min()
                max_value = self.train_df[feature].max()

                if self.permitted_range[feature][0] < min_value and self.permitted_range[feature][1] > max_value:
                    return False
            else:
                self.permitted_range[feature] = [self.train_df[feature].min(), self.train_df[feature].max()]
        return True

    def get_features_range(self):
        ranges = {}
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                self.train_df[feature_name].min(), self.train_df[feature_name].max()]
        return ranges

    def get_data_type(self, col):
        """Infers data type of a feature from the training data."""
        if((self.data_df[col].dtype == np.int64) or (self.data_df[col].dtype == np.int32)):
            return 'int'
        elif((self.data_df[col].dtype == np.float64) or (self.data_df[col].dtype == np.float32)):
            return 'float'
        else:
            raise ValueError("Unknown data type of feature %s: must be int or float" %col)

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
        return result

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()
            result[feature_name] = (
                df[feature_name]*(max_value - min_value)) + min_value
        return result

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0]*len(self.encoded_feature_names)])
        maxx = np.array([[1.0]*len(self.encoded_feature_names)])

        for idx, feature_name in enumerate(self.continuous_feature_names):
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()

            if normalized:
                minx[0][idx] = (self.permitted_range[feature_name]
                                [0] - min_value) / (max_value - min_value)
                maxx[0][idx] = (self.permitted_range[feature_name]
                                [1] - min_value) / (max_value - min_value)
            else:
                minx[0][idx] = self.permitted_range[feature_name][0]
                maxx[0][idx] = self.permitted_range[feature_name][1]
        return minx, maxx

    def split_data(self, data):
        train_df, test_df = train_test_split(
            data, test_size=self.test_size, random_state=self.test_split_random_state)
        return train_df, test_df

    def get_mads(self, normalized=False):
        """Computes Median Absolute Deviation of features."""

        mads = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(self.train_df[feature].values - np.median(self.train_df[feature].values)))
        else:
            normalized_train_df = self.normalize_data(self.train_df)
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
                    abs(list(set(self.train_df[feature].tolist())) - np.median(list(set(self.train_df[feature].tolist())))), quantile)
        else:
            normalized_train_df = self.normalize_data(self.train_df)
            for feature in self.continuous_feature_names:
                quantiles[feature] = np.quantile(
                    abs(list(set(normalized_train_df[feature].tolist())) - np.median(list(set(normalized_train_df[feature].tolist())))), quantile)
        return quantiles

    def get_data_params(self):
        """Gets all data related params for DiCE."""

        minx, maxx = self.get_minx_maxx(normalized=True)

        # get the column indexes of categorical features after one-hot-encoding
        self.encoded_categorical_feature_indexes = self.get_encoded_categorical_feature_indexes()

        return minx, maxx, self.encoded_categorical_feature_indexes

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
        if features_to_vary == "all":
            return [i for i in range(len(self.encoded_feature_names))]
        else:
            ixs = []
            encoded_cats_ixs = self.get_encoded_categorical_feature_indexes()
            for colidx, col in enumerate(self.encoded_feature_names):
                if colidx in encoded_cats_ixs and col.startswith(tuple(features_to_vary)):
                    ixs.append(colidx)
                elif colidx not in encoded_cats_ixs and col in features_to_vary:
                    ixs.appen(colidx)
            return ixs

    def from_dummies(self, data, prefix_sep='_'):
        """Gets the original data from dummy encoded data with k levels."""
        out = data.copy()
        for l in self.categorical_feature_names:
            cols, labs = [[c.replace(
                x, "") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
            out[l] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decimal_precisions(self):
        """"Gets the precision of continuous features in the data."""
        # if the precision of a continuous feature is not given, we use the maximum precision of the modes to capture the precision of majority of values in the column.
        precisions = [0]*len(self.feature_names)
        for ix, col in enumerate(self.continuous_feature_names):
            if((self.continuous_features_precision is not None) and (col in self.continuous_features_precision)):
                precisions[ix] = self.continuous_features_precision[col]
            elif((self.data_df[col].dtype == np.float32) or (self.data_df[col].dtype == np.float64)):
                modes = self.data_df[col].mode()
                maxp = len(str(modes[0]).split('.')[1]) # maxp stores the maximum precision of the modes
                for mx in range(len(modes)):
                    prec = len(str(modes[mx]).split('.')[1])
                    if prec > maxp:
                        maxp = prec
                maxp = min(maxp, 4)
                precisions[ix] = maxp
        return precisions

    def get_decoded_data(self, data):
        """Gets the original data from dummy encoded data."""
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(data=data, index=index,
                                columns=self.encoded_feature_names)
        return self.from_dummies(data)

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

    def prepare_query_instance(self, query_instance, encode):
        """Prepares user defined test input for DiCE."""

        if isinstance(query_instance, list):
            query_instance = {'row1': query_instance}
            test = pd.DataFrame.from_dict(
                query_instance, orient='index', columns=self.feature_names)

        elif isinstance(query_instance, dict):
            query_instance = dict(zip(query_instance.keys(), [[q] for q in query_instance.values()]))
            test = pd.DataFrame(query_instance, columns=self.feature_names)

        test = test.reset_index(drop=True)

        if encode is False:
            return self.normalize_data(test)
        else:
            temp = self.prepare_df_for_encoding()

            temp = temp.append(test, ignore_index=True, sort=False)
            temp = self.one_hot_encode_data(temp)
            temp = self.normalize_data(temp)

            return temp.tail(test.shape[0]).reset_index(drop=True)

    def get_dev_data(self, model_interface, desired_class, filter_threshold=0.5):
        """Constructs dev data by extracting part of the test data for which finding counterfactuals make sense."""

        # create TensorFLow session if one is not already created
        if tf.get_default_session() is not None:
            self.data_sess = tf.get_default_session()
        else:
            self.data_sess = tf.InteractiveSession()

        # loading trained model
        model_interface.load_model()

        # get the permitted range of change for each feature
        minx, maxx = self.get_minx_maxx(normalized=True)

        # get the transformed data: continuous features are normalized to fall in the range [0,1], and categorical features are one-hot encoded
        data_df_transformed = self.normalize_data(self.one_hot_encoded_data)

        # split data - nomralization considers only train df and there is no leakage due to transformation before train-test splitting
        _, test = self.split_data(data_df_transformed)
        test = test.drop_duplicates(
            subset=self.encoded_feature_names).reset_index(drop=True)

        # finding target predicted probabilities
        input_tensor = tf.Variable(minx, dtype=tf.float32)
        output_tensor = model_interface.get_output(
            input_tensor)  # model(input_tensor)
        temp_data = test[self.encoded_feature_names].values.astype(np.float32)
        dev_preds = [self.data_sess.run(output_tensor, feed_dict={
                                        input_tensor: np.array([dt])}) for dt in temp_data]
        dev_preds = [dev_preds[i][0][0] for i in range(len(dev_preds))]

        # filtering examples which have predicted value >/< threshold
        dev_data = test[self.encoded_feature_names]
        if desired_class == 0:
            idxs = [i for i in range(len(dev_preds))
                    if dev_preds[i] > filter_threshold]
        else:
            idxs = [i for i in range(len(dev_preds))
                    if dev_preds[i] < filter_threshold]
        dev_data = dev_data.iloc[idxs]
        dev_preds = [dev_preds[i] for i in idxs]

        # convert from one-hot encoded vals to user interpretable fromat
        dev_data = self.from_dummies(dev_data)
        dev_data = self.de_normalize_data(dev_data)
        return dev_data[self.feature_names], dev_preds  # values.tolist()
