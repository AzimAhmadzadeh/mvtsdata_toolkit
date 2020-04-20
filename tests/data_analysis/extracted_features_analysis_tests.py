import unittest
import CONSTANTS as CONST
from mvtsdatatoolkit.data_analysis import extracted_features_analysis
import pandas as pd
import os


class TestExtractedFeaturesAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        f_path = os.path.join(CONST.ROOT, 'tests/test_dataset/non_unittest_extracted_features.csv')
        cls.mvts_df = pd.read_csv(f_path, sep='\t')
        cls.efa = extracted_features_analysis.ExtractedFeaturesAnalysis(cls.mvts_df, ['id'])
        cls.efa.compute_summary()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_class_population(self):
        expected_class_population = pd.DataFrame(
            {'lab': ['N', 'C', 'M', 'X', 'B'], 'Population': [2222, 335, 115, 16, 10]})
        actual_class_population = self.efa.get_class_population(label='lab')

        self.assertEqual(expected_class_population.equals(actual_class_population), True,
                         'Expected != Actual')

    def test_get_missing_values(self):
        expected_missing_vals = pd.DataFrame(
            {'Feature-Name': ['TOTUSJH_min', 'TOTUSJH_max', 'TOTUSJH_median'],
             'Null-Count': [1, 1, 2]})

        actual_missing_vals = self.efa.get_missing_values()
        actual_missing_vals = actual_missing_vals.iloc[:3]

        self.assertEqual(expected_missing_vals.equals(actual_missing_vals), True,
                         'Expected != Actual')

    def test_1x_get_five_num_summary(self):
        feature_name = 'TOTUSJH_median'
        expected_values = [749.4241793039122, 1350.5150940961726, 1.490240826139002,
                           39.78626597997699,
                           180.54826356053326, 798.9918748307275, 10634.326797965652]
        df_summary = self.efa.get_five_num_summary()
        actual_values = df_summary[df_summary['Feature-Name'] == feature_name].values
        actual_values = list(actual_values[0, 1:])
        self.assertListEqual(actual_values, expected_values, 'Expected != Actual')

    def test_2x_get_five_num_summary(self):
        expected_colnames = ['Feature-Name', 'mean', 'std', 'min', '25th', '50th', '75th', 'max']
        df_summary = self.efa.get_five_num_summary()
        actual_colnames = list(df_summary)
        self.assertListEqual(actual_colnames, expected_colnames, 'Expected != Actual')


if __name__ == '__main__':
    unittest.main()
