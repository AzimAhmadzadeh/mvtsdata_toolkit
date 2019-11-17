import unittest
import CONSTANTS as CONST
from data_analysis import extracted_features_analysis
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
            {'Label': ['N', 'C', 'M', 'X', 'B'], 'Count': [2222, 335, 115, 16, 10]})
        expected_class_population = expected_class_population.set_index('Label')
        actual_class_population = self.efa.get_class_population(label='lab')
        self.assertEqual(expected_class_population.equals(actual_class_population), True,
                         'Expected != Actual')

    def test_get_missing_values(self):
        expected_class_population = pd.DataFrame(
            {'Feature Name': ['TOTUSJH_min', 'TOTUSJH_max', 'TOTUSJH_median'],
             'Null Count': [1, 1, 2]})

        actual_class_population = self.efa.get_missing_values()
        actual_class_population = actual_class_population.iloc[:3]
        self.assertEqual(expected_class_population.equals(actual_class_population), True,
                         'Expected != Actual')


if __name__ == '__main__':
    unittest.main()
