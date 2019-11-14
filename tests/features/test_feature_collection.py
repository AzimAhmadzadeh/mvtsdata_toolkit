import unittest
import pandas as pd
import numpy as np
import features.feature_collection as ftr_col


class TestFeatureCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        input_data = np.array([1234,12.34,23.44,456.23,45,22.22])
        cls.input_series = pd.Series(input_data)
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_min(self):
        expected_value = 12.34
        actual_value = ftr_col.get_min(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_max(self):
        expected_value = 1234.0
        actual_value = ftr_col.get_max(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_median(self):
        expected_value = 34.22
        actual_value = ftr_col.get_median(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_average_absolute_change(self):
        expected_value = 419.912
        actual_value = ftr_col.get_average_absolute_change(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_average_absolute_derivative_change(self):
        expected_value = 721.73
        actual_value = ftr_col.get_average_absolute_derivative_change(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_avg_mono_decrease_slope(self):
        expected_value = -377.75
        actual_value = ftr_col.get_avg_mono_decrease_slope(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_avg_mono_increase_slope(self):
        expected_value = 148
        actual_value = round(ftr_col.get_avg_mono_increase_slope(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_dderivative_kurtosis(self):
        expected_value = -1
        actual_value = round(ftr_col.get_dderivative_kurtosis(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_dderivative_mean(self):
        expected_value = -242
        actual_value = round(ftr_col.get_dderivative_mean(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_dderivative_skewness(self):
        expected_value = -1
        actual_value = round(ftr_col.get_dderivative_skewness(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_difference_of_maxs(self):
        expected_value = 777.77
        actual_value = ftr_col.get_difference_of_maxs(self.input_series)
        self.assertEqual(expected_value,actual_value,'Expected != Actual')


    def test_get_difference_of_means(self):
        expected_value = 249
        actual_value = round(ftr_col.get_difference_of_means(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')

    def test_get_difference_of_means2(self):
        expected_value = 22
        actual_value = round(ftr_col.get_difference_of_medians(self.input_series))
        self.assertEqual(expected_value,actual_value,'Expected != Actual')


if __name__ == '__main__':
    unittest.main()
