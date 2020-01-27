import unittest
from data_analysis.mvts_data_analysis import MVTSDataAnalysis
import numpy as np
import pandas as pd


class TestMVTSDataAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        f_path = 'tests/configs/feature_extraction_configs.yml'
        cls.mvts = MVTSDataAnalysis(f_path)
        cls.mvts.compute_summary(params_name=['TOTUSJH', 'TOTBSQ', 'TOTPOT'])

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_computing_null_counts(self):
        actual_counts = self.mvts.summary.loc[:, 'Null-Count'].tolist()
        # the expected counts are verified by LibreCalc using 'COUNTBLANK` function.
        expected_counts = [4, 2, 2]
        np.testing.assert_equal(actual_counts, expected_counts)

    def test_computing_val_counts(self):
        actual_counts = self.mvts.summary.loc[:, 'Val-Count'].tolist()
        # the expected counts are verified by LibreCalc using 'COUNTA` function.
        expected_counts = [716, 718, 718]
        np.testing.assert_equal(actual_counts, expected_counts)

    def test_get_number_of_mvts(self):
        expected_count = 12
        actual_count = self.mvts.get_number_of_mvts()
        self.assertEqual(actual_count, expected_count)

    def test_computing_mean(self):
        accepted_err_rate = 0.1  # i.e., 10% of the expected values
        actual_means = self.mvts.summary.loc[:, 'mean'].tolist()

        # the expected values are verified by what LibreCalc gives using 'MEANA' function.
        expected_means = [3000.64, 40492563413.12, 7.935093000765689e+23]

        accepted_diff = np.abs(np.multiply(expected_means, accepted_err_rate))
        actual_diff = np.absolute(np.subtract(actual_means, expected_means))

        self.assertListEqual(list(actual_diff < accepted_diff), [True, True, True])
        # np.testing.assert_almost_equal(actual_means, expected_means, decimal=2)

    def test_computing_min(self):
        actual_mins = self.mvts.summary.loc[:, 'min'].tolist()
        # the expected mins are verified by what LibreCalc gives using 'MINA' function.
        expected_mins = [7.463776684381733, 51309673.6747, 3.718513952226165e+20]
        np.testing.assert_almost_equal(actual_mins, expected_mins, decimal=4)

    def test_computing_max(self):
        actual_maxs = self.mvts.summary.loc[:, 'max'].tolist()
        # the expected maxs are verified by what LibreCalc gives using 'MAXA' function.
        expected_maxs = [10339.487887200366, 138681554867.8218, 3.215687280392037e+24]
        np.testing.assert_almost_equal(actual_maxs, expected_maxs, decimal=4)

    def test_computing_q1(self):
        accepted_err_rate = 0.3  # i.e., 30% of the expected values
        actual_q1s = np.array(self.mvts.summary.loc[:, '25th'])

        # the expected Q1's are verified by what LibreCalc gives using 'PERCENTILE' function.
        # Note: This is a very loose test. The discrepancy is quite large in this test-case because
        # of the sample size; the estimation of Q1 (using tDigest) works poorly on a small number
        # of mvts, and here we have only 12 files. The accurate results, calculated
        # without any estimation, should be as follows:
        # [991.552229323097, 10549394315.9591, 1.3640682946706E+023]
        expected_q1s = np.array([991.552229323097, 10549394315.9591, 1.3640682946706E+023])
        accepted_diff = np.abs(np.multiply(expected_q1s, accepted_err_rate))
        actual_diff = np.absolute(np.subtract(actual_q1s, expected_q1s))
        self.assertListEqual(list(actual_diff < accepted_diff), [True, True, True])

    def test_computing_q2(self):
        accepted_err_rate = 0.12  # i.e., 12% of the expected values
        actual_q2s = np.array(self.mvts.summary.loc[:, '50th'])

        # the expected medians are verified by what LibreCalc gives using 'PERCENTILE' function.
        # Note: This is a very loose test. The discrepancy is quite large in this test-case because
        # of the sample size; the estimation of median (using tDigest) works poorly on a small
        # number of mvts, and here we have only 12 files. The accurate results, calculated
        # without any estimation, should be as follows:
        # [1916.85297388505, 21941909770.4986, 3.58870685915509E+023]
        expected_q2s = np.array([1916.85297388505, 21941909770.4986, 3.58870685915509E+023])
        accepted_diff = np.abs(np.multiply(expected_q2s, accepted_err_rate))
        actual_diff = np.absolute(np.subtract(actual_q2s, expected_q2s))
        self.assertListEqual(list(actual_diff < accepted_diff), [True, True, True])

    def test_computing_q3(self):
        accepted_err_rate = 0.12  # i.e., 12% of the expected values
        actual_q3s = np.array(self.mvts.summary.loc[:, '75th'])

        # the expected Q3's are verified by what LibreCalc gives using 'PERCENTILE' function.
        # Note: This is a very loose test. The discrepancy is quite large in this test-case because
        # of the sample size; the estimation of Q3 (using tDigest) works poorly on a small number
        # of mvts, and here we have only 12 files. The accurate results, calculated without any
        # estimation, should be as follows:
        # [3595.09630331655, 58144661605.687, 8.8219110661967E+023]
        expected_q3s = np.array([3595.09630331655, 58144661605.687, 8.8219110661967E+023])
        accepted_diff = np.abs(np.multiply(expected_q3s, accepted_err_rate))
        actual_diff = np.absolute(np.subtract(actual_q3s, expected_q3s))
        self.assertListEqual(list(actual_diff < accepted_diff), [True, True, True])

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
        """ Tests if proper exception will be raised if `compute_summary` has not been executed."""
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
        """ Tests if proper exception will be raised if `compute_summary` has not been executed."""
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
        """ Tests if proper exception will be raised if `compute_summary` has not been executed."""
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


if __name__ == '__main__':
    unittest.main()
