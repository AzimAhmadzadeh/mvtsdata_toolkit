import unittest
import os
import pandas as pd
import numpy as np
import CONSTANTS as CONST
import normalizing.normalizer as normalizer
import warnings


class TestNormalizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        path_to_mvts = os.path.join(CONST.ROOT,
                                    'tests/test_dataset/extracted_features'
                                    '/extracted_features_TEST_NORMALIZER.csv')
        cls.df = pd.read_csv(path_to_mvts, sep='\t')

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_zero_one_normalize_X1(self):
        """ Tests if this method keeps non-numeric columns."""
        df_normalized = normalizer.zero_one_normalize(self.df)
        colnames_before = list(self.df)
        colnames_after = list(df_normalized)
        self.assertListEqual(colnames_after, colnames_before)

    def test_zero_one_normalize_X2(self):
        """ Tests the zero-one normalization's results against a quick normalization of one
        single column."""
        # take one column, from the raw data, as an example to test
        arr_raw = np.array(self.df.loc[:, 'TOTUSJH_mean'])
        df_normalized = normalizer.zero_one_normalize(self.df)
        # take the same column after normalization
        arr_normalized = np.array(df_normalized.loc[:, 'TOTUSJH_mean'])

        # manually normalize the values
        arr_min = np.nanmin(arr_raw)
        arr_max = np.nanmax(arr_raw)
        arr_expected = (arr_raw - arr_min) / (arr_max - arr_min)
        # Verify if they are equal (in 10 decimal places)
        np.testing.assert_array_almost_equal(arr_expected, arr_normalized, decimal=10)

    def test_negativeone_one_normalize_X1(self):
        """ Test if this method keeps non-numeric columns."""
        df_normalized = normalizer.negativeone_one_normalize(self.df)
        colnames_before = list(self.df)
        colnames_after = list(df_normalized)
        self.assertListEqual(colnames_after, colnames_before)

    def test_negativeone_one_normalize_X2(self):
        """ Tests the (-1, +1)-normalization's results against a quick normalization of one
        single column."""
        # take one column, from the raw data, as an example to test
        arr_raw = np.array(self.df.loc[:, 'TOTUSJH_mean'])
        df_normalized = normalizer.negativeone_one_normalize(self.df)
        # take the same column after normalization
        arr_normalized = np.array(df_normalized.loc[:, 'TOTUSJH_mean'])

        # manually normalize the values
        arr_min = np.nanmin(arr_raw)
        arr_max = np.nanmax(arr_raw)
        scale = 2 / (arr_max - arr_min)
        arr_expected = scale * arr_raw - 1 - arr_min * scale
        # Verify if they are equal (in 10 decimal places)
        np.testing.assert_array_almost_equal(arr_expected, arr_normalized, decimal=10)

    def test_standardize_X1(self):
        """ Test if this method keeps non-numeric columns."""
        df_normalized = normalizer.standardize(self.df)
        colnames_before = list(self.df)
        colnames_after = list(df_normalized)
        self.assertListEqual(colnames_after, colnames_before)

    def test_standardize_X2(self):
        """ Tests the standardization's results against a quick normalization of one
        single column."""
        # take one column, from the raw data, as an example to test
        arr_raw = np.array(self.df.loc[:, 'TOTUSJH_mean'])
        df_normalized = normalizer.standardize(self.df)
        # take the same column after normalization
        arr_normalized = np.array(df_normalized.loc[:, 'TOTUSJH_mean'])

        # manually normalize the values
        mu = np.nanmean(arr_raw)
        std = np.std(arr_raw)
        arr_expected = (arr_raw - mu) / std
        # Verify if they are equal (in 10 decimal places)
        np.testing.assert_array_almost_equal(arr_expected, arr_normalized, decimal=10)

    def test_robust_standardize_X1(self):
        """ Test if this method keeps non-numeric columns."""
        df_normalized = normalizer.robust_standardize(self.df)
        colnames_before = list(self.df)
        colnames_after = list(df_normalized)
        self.assertListEqual(colnames_after, colnames_before)


if __name__ == '__main__':
    unittest.main()
