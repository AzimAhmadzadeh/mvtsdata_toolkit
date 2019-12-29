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
        expected_size = 40234.166  # bytes
        actual_size = self.mvts.get_average_mvts_size()
        self.assertAlmostEqual(actual_size, expected_size, places=2)

    def test_get_total_mvts_size(self):
        expected_size = 482810  # bytes
        actual_size = self.mvts.get_total_mvts_size()
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

    def test_1x_get_six_num_summary(self):
        expected_colnames = ['Parameter-Name', 'mean', 'min', '25th', '50th', '75th', 'max']
        actual_colnames = list(self.mvts.get_six_num_summary())
        self.assertListEqual(actual_colnames, expected_colnames)

    def test_2x_get_six_num_summary(self):
        """ Test if proper exception will be raised if `compute_summary` has not been executed."""
        _backup = self.mvts.summary.copy()
        self.mvts.summary = pd.DataFrame()  # summary is an empty dataframe now
        with self.assertRaises(ValueError) as context:
            self.mvts.get_six_num_summary()
            self.assertTrue(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                getting the 6-num summary.
                '''
                in context.exception)

        self.mvts.summary = _backup.copy()  # this is needed for the rest of the tests.

    def test_print_summary(self):
        """ Test if proper exception will be raised if `compute_summary` has not been executed."""
        _backup = self.mvts.summary.copy()
        self.mvts.summary = pd.DataFrame()  # summary is an empty dataframe now
        with self.assertRaises(ValueError) as context:
            self.mvts.print_summary()
            self.assertTrue(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                printing the results.
                '''
                in context.exception)

        self.mvts.summary = _backup.copy()  # this is needed for the rest of the tests.

    def test_summary_to_scv(self):
        """ Test if proper exception will be raised if `compute_summary` has not been executed."""
        _backup = self.mvts.summary.copy()
        self.mvts.summary = pd.DataFrame()  # summary is an empty dataframe now
        with self.assertRaises(ValueError) as context:
            self.mvts.summary_to_csv(output_path='xxx/', file_name='yyy.csv')
            self.assertTrue(
                '''
                The summary is empty. The method `compute_summary` needs to be executed before 
                saving it as a csv file.
                '''
                in context.exception)
        self.mvts.summary = _backup.copy()  # this is needed for the rest of the tests.

    def test_x(self):
        print(self.mvts.summary)


if __name__ == '__main__':
    unittest.main()
