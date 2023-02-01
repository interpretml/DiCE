import pytest

import dice_ml


def test_data_initiation(public_data_object, private_data_object):
    assert isinstance(public_data_object, dice_ml.data_interfaces.public_data_interface.PublicData), \
        "the given parameters should instantiate PublicData class"

    assert isinstance(private_data_object, dice_ml.data_interfaces.private_data_interface.PrivateData), \
        "the given parameters should instantiate PrivateData class"


class TestCommonDataMethods:
    """
    Test methods common to Public and Private data interfaces modules.
    """
    @pytest.fixture(autouse=True)
    def _get_data_object(self, public_data_object, private_data_object):
        self.d = [public_data_object, private_data_object]

    def test_get_valid_mads(self):
        # public data
        for normalized in [True, False]:
            mads = self.d[0].get_valid_mads(normalized=normalized, display_warnings=False, return_mads=True)
            # mads denotes variability in features and should be positive for DiCE.
            assert all(mads[feature] > 0 for feature in mads)

            if normalized:
                min_value = 0
                max_value = 1

            errors = 0
            for feature in mads:
                if not normalized:
                    min_value = self.d[0].data_df[feature].min()
                    max_value = self.d[0].data_df[feature].max()

                if mads[feature] > max_value - min_value:
                    errors += 1
            assert errors == 0  # mads can't be larger than the feature range

        # private data
        for normalized in [True, False]:
            mads = self.d[1].get_valid_mads(normalized=normalized, display_warnings=False, return_mads=True)
            # no mad is provided for private data by default, so a practical alternative is keeping all value at 1.
            # Check get_valid_mads() in data interface classes for more info.
            assert all(mads[feature] == 1 for feature in mads)

    # @pytest.mark.parametrize(
    # "encode_categorical, output_query",
    # [('label', [0.068, (2/3), (3/7), (3/4), (4/5), 1.0, 0.0, 0.449]),
    # ('one-hot', [0.068, 0.449, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    #              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])]
    # )
    # def test_prepare_query_instance(self, sample_adultincome_query, encode_categorical, output_query):
    #     """
    #     Tests prepare_query_instance method that covnerts continuous features into [0,1] range and
    #     one-hot encodes categorical features.
    #     """
    #     for d in self.d:
    #         prepared_query = d.prepare_query_instance(
    #             query_instance=sample_adultincome_query, encoding=encode_categorical).iloc[0].tolist()
    #         assert output_query == pytest.approx(prepared_query, abs=1e-3)

    # TODO: add separate test method for label-encoded data
    def test_ohe_min_max_transformed_query_instance(self, sample_adultincome_query):
        """
        Tests method that converts continuous features into [0,1] range and one-hot encodes categorical features.
        """
        output_query = [0.068, 0.449, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        d = self.d[0]
        prepared_query = d.get_ohe_min_max_normalized_data(query_instance=sample_adultincome_query).iloc[0].tolist()
        assert output_query == pytest.approx(prepared_query, abs=1e-3)

    def test_encoded_categorical_features(self):
        """
        Tests if correct encoded categorical feature indexes are returned. Should work even when feature names
        are starting with common names.
        TODO also test for private data interface
        """
        res = []
        d = self.d[0]
        # for d in self.d:
        #     d.categorical_feature_names = ['cat1', 'cat2']
        #     d.continuous_feature_names = ['cat2_cont1', 'cont2']
        #     d.encoded_feature_names = ['cat2_cont1', 'cont2', 'cat1_val1', 'cat1_val2', 'cat2_val1', 'cat2_val2']
        print(d.data_df)
        temp_ohe_data = d.get_ohe_min_max_normalized_data(d.data_df.iloc[[0]])
        d.create_ohe_params(temp_ohe_data)
        res.append(d.get_encoded_categorical_feature_indexes())
        assert [2, 3, 4, 5] == res[0][0]  # there are 4 types of workclass
        assert len(res[0][1]) == 8  # eight types of education
        assert len(res[0][-1]) == 2  # two types of gender in the data
        # 2,3,4,5 are correct indexes of encoded categorical features and the data object's method should not
        # return the first continuous feature that starts with the same name. Returned value should be a list of lists.

    def test_features_to_vary(self):
        """
        Tests if correct indexes of features are returned. Should work even when feature names are starting with common names.

        TODO: also make it work for private_data_interface
        """
        res = []
        d = self.d[0]
        temp_ohe_data = d.get_ohe_min_max_normalized_data(d.data_df.iloc[[0]])
        d.create_ohe_params(temp_ohe_data)
        # d.create_ohe_params()
        # d.categorical_feature_names = ['cat1', 'cat2']
        # d.encoded_feature_names = ['cat2_cont1', 'cont2', 'cat1_val1', 'cat1_val2', 'cat2_val1', 'cat2_val2']
        # d.continuous_feature_names = ['cat2_cont1', 'cont2']
        res.append(d.get_indexes_of_features_to_vary(features_to_vary=['workclass']))
        # 4,5 are correct indexes of features that can be varied and the data object's method should not return
        # the first continuous feature that starts with the same name.
        assert [2, 3, 4, 5] == res[0]
