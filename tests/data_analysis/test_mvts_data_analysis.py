import unittest
from data_analysis.mvts_data_analysis import MVTSDataAnalysis
import numpy as np
import pandas as pd


class TestMVTSDataAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        f_path = 'tests/configs/feature_extraction_configs.yml'
        cls.mvts = MVTSDataAnalysis(f_path)
        cls.mvts.compute_summary()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_number_of_mvts(self):
        expected_count = 12
        actual_count = self.mvts.get_number_of_mvts()
        self.assertEqual(actual_count, expected_count)

    def test_get_average_mvts_size(self):
        expected_size = 40234.166
        actual_size = self.mvts.get_average_mvts_size()
        self.assertAlmostEqual(actual_size, expected_size, places=2)

    def test_get_missing_values(self):
        expected_class_population = pd.DataFrame({
            'Parameter-Name': ['TOTUSJH', 'TOTBSQ', 'TOTPOT'],
            'Null-Count': [4, 2, 2]
        }, index=[0, 1, 2], dtype=object)
        actual_class_population = self.mvts.get_missing_values()
        actual_class_population = actual_class_population.iloc[:3]
        pd.testing.assert_frame_equal(actual_class_population, expected_class_population)
        pd.testing.assert_series_equal(actual_class_population['Null-Count'],
                                       expected_class_population['Null-Count'])


if __name__ == '__main__':
    unittest.main()
