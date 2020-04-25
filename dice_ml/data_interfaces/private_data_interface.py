"""Module containing meta data information about private data."""

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.NOTSET)

class PrivateData:
    """A data interface for private data with meta information."""

    def __init__(self, params):
        """Init method

        :param features: Dictionary with feature names as keys and range in int/float (for continuous features) or categories in string (for categorical features) as values.
        :param outcome_name: Outcome feature name.
        :param type_and_precision (optional): Dictionary with continuous feature names as keys. If the feature is of type int, just string 'int' should be provided, if the feature is of type float, a list of type and precision should be provided. For instance, type_and_precision: {cont_f1: 'int', cont_f2: ['float', 2]} for continuous features cont_f1 and cont_f2 of type int and float (and precision up to 2 decimal places) respectively. Default value is None and all features are treated as int.
        :param mad (optional): Dictionary with feature names as keys and corresponding Median Absolute Deviations (MAD) as values. Default MAD value is 1 for all features.

        """

        if type(params['features']) is dict:
            features_dict = params['features']
        else:
            raise ValueError(
                "should provide dictionary with feature names as keys and range (for continuous features) or categories (for categorical features) as values")

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature")

        if 'type_and_precision' in params:
            self.type_and_precision = params['type_and_precision']
        else:
            self.type_and_precision = {}

        self.continuous_feature_names = []
        self.permitted_range = {}
        self.categorical_feature_names = []
        self.categorical_levels = {}

        for feature in features_dict:
            if type(features_dict[feature][0]) is int:  # continuous feature
                self.continuous_feature_names.append(feature)
                self.permitted_range[feature] = features_dict[feature]
            else:
                self.categorical_feature_names.append(feature)
                self.categorical_levels[feature] = features_dict[feature]

        if 'mad' in params:
            self.mad = params['mad']
        else:
            self.mad = {}

        # self.continuous_feature_names + self.categorical_feature_names
        self.feature_names = list(features_dict.keys())

        self.continuous_feature_indexes = [list(features_dict.keys()).index(
            name) for name in self.continuous_feature_names if name in features_dict]

        self.categorical_feature_indexes = [list(features_dict.keys()).index(
            name) for name in self.categorical_feature_names if name in features_dict]

        if len(self.categorical_feature_names) > 0:
            # simulating sklearn's one-hot-encoding
            # continuous features on the left
            self.encoded_feature_names = [
                feature for feature in self.continuous_feature_names]
            for feature_name in self.categorical_feature_names:
                for category in sorted(self.categorical_levels[feature_name]):
                    self.encoded_feature_names.append(
                        feature_name+'_'+category)

        for feature_name in self.continuous_feature_names:
            if feature_name not in self.type_and_precision:
                self.type_and_precision[feature_name] = 'int'

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.permitted_range[feature_name][1]
            min_value = self.permitted_range[feature_name][0]
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
        return result

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.permitted_range[feature_name][1]
            min_value = self.permitted_range[feature_name][0]
            result[feature_name] = (
                df[feature_name]*(max_value - min_value)) + min_value
        return result

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0]*len(self.encoded_feature_names)])
        maxx = np.array([[1.0]*len(self.encoded_feature_names)])

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
                logging.info(" MAD is not given for feature %s, so using 1.0 as MAD instead.", feature)

        if return_mads:
            return mads

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
                col) for col in self.encoded_feature_names if col.startswith(col_parent)]
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
        precisions = [0]*len(self.feature_names)
        for ix, feature_name in enumerate(self.continuous_feature_names):
            type_prec = self.type_and_precision[feature_name]
            if type_prec == 'int':
                precisions[ix] = 0
            else:
                precisions[ix] = self.type_and_precision[feature_name][1]
        return precisions

    def get_decoded_data(self, data):
        """Gets the original data from dummy encoded data."""
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(data=data, index=index,
                                columns=self.encoded_feature_names)
        return self.from_dummies(data)

    def prepare_df_for_encoding(self):
        """Facilitates get_test_inputs() function."""
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(self.categorical_levels[cat_feature])

        df = pd.DataFrame({colnames[0]: levels[0]})
        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = self.continuous_feature_names
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        return df

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(data, drop_first=False, columns=self.categorical_feature_names)

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
        raise ValueError(
            "Cannot compute dev data from only meta data information")
