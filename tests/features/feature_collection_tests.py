import unittest
import pandas as pd
import numpy as np
from features import feature_collection as fc


class TestFeatureCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        input_arr = np.array(
            [0.719, 0.576, 0.5, 0.356, 0.232, 0.113, 0.053, 0.032, 0.063, 0.255, 0.248, 0.34, 0.393,
             0.461, 0.693, 0.826, 0.955, 0.813, 0.979, 0.995, 0.9, 1., 0.966, 0.976, 0.857, 0.759,
             0.561, 0.567, 0.534, 0.453, 0.443, 0.304, 0.308, 0.215, 0.261, 0.24, 0.23, 0.259,
             0.225, 0.17, 0.08, 0.011, 0., 0.091, 0.051, 0.03, 0.05, 0.086, 0.136, 0.191, 0.233,
             0.304, 0.332, 0.366, 0.541, 0.658, 0.71, 0.859, 0.977, 0.993])
        cls.input_series = pd.Series(input_arr)
        # input_data = np.array([1234, 12.34, 23.44, 456.23, 45, 22.22])
        # cls.input_series = pd.Series(input_data)

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_min(self):
        expected_value = np.min(self.input_series)
        actual_value = fc.get_min(self.input_series)
        self.assertEqual(expected_value, actual_value, 'Expected != Actual')

    def test_get_max(self):
        expected_value = np.max(self.input_series)
        actual_value = fc.get_max(self.input_series)
        self.assertEqual(expected_value, actual_value, 'Expected != Actual')

    def test_get_median(self):
        expected_value = np.median(self.input_series)
        actual_value = fc.get_median(self.input_series)
        self.assertEqual(expected_value, actual_value, 'Expected != Actual')

    def test_1X_difference_derivative(self):
        """ When gap=1, the i-th value of the output should be input_series[i] - input_series[
        i-1]."""
        gap = 1
        expected_value = np.subtract(self.input_series[gap:], self.input_series[:-gap])
        actual_value = fc._difference_derivative(self.input_series, gap)
        self.assertListEqual(list(expected_value), list(actual_value))

    def test_2X_difference_derivative(self):
        """ When gap=1, the length of the output should be 59."""
        gap = 1
        expected_value = len(self.input_series) - 1
        actual_value = len(fc._difference_derivative(self.input_series, gap))
        self.assertEqual(expected_value, actual_value)

    def test_3X_difference_derivative(self):
        """ When gap=60, the output should be an empty list."""
        gap = len(self.input_series)
        expected_value = []
        actual_value = fc._difference_derivative(self.input_series, gap)
        self.assertListEqual(list(expected_value), list(actual_value))

    def test_4X_difference_derivative(self):
        """ When gap=0, the output should be None."""
        gap = 0
        expected_value = None
        actual_value = fc._difference_derivative(self.input_series, gap)
        self.assertEqual(expected_value, actual_value)

    def test_5X_difference_derivative(self):
        """ When gap<0, the output should be None."""
        gap = -10
        expected_value = None
        actual_value = fc._difference_derivative(self.input_series, gap)
        self.assertEqual(expected_value, actual_value)

    def test_get_average_absolute_change(self):
        expected_value = 0.07539
        actual_value = fc.get_average_absolute_change(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_average_absolute_derivative_change(self):
        expected_value = 0.08019
        actual_value = fc.get_average_absolute_derivative_change(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_avg_mono_decrease_slope(self):
        expected_value = -0.04560
        actual_value = fc.get_avg_mono_decrease_slope(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_avg_mono_increase_slope(self):
        expected_value = 0.04029
        actual_value = fc.get_avg_mono_increase_slope(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_dderivative_kurtosis(self):
        expected_value = -0.41472
        actual_value = fc.get_dderivative_kurtosis(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_dderivative_mean(self):
        expected_value = 0.00464
        actual_value = fc.get_dderivative_mean(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_dderivative_skewness(self):
        expected_value = 0.15912
        actual_value = fc.get_dderivative_skewness(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_difference_of_maxs(self):
        expected_value = 0.00700
        actual_value = fc.get_difference_of_maxs(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_difference_of_means(self):
        expected_value = 0.26070
        actual_value = fc.get_difference_of_means(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)

    def test_get_difference_of_medians(self):
        expected_value = 0.32750
        actual_value = fc.get_difference_of_medians(self.input_series)
        self.assertAlmostEqual(expected_value, actual_value, places=5)


if __name__ == '__main__':
    unittest.main()
