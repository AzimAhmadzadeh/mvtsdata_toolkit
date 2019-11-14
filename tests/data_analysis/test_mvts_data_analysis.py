import unittest
import CONSTANTS as CONST
from data_analysis import mvts_data_analysis
import pandas as pd
import os


class TestMVTSDataAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        f_path = os.path.join(CONST.ROOT, 'tests/test_dataset/mvts')

        cls.mvts = mvts_data_analysis.MVTSDataAnalysis(f_path)
        cls.mvts.compute_summary()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_missing_values(self):
        expected_class_population = pd.DataFrame(
            {'Feature Name': ['TOTUSJH', 'TOTBSQ', 'TOTPOT'], 'Null Count': [1, 1, 1]})

        actual_class_population = self.mvts.get_missing_values()

        actual_class_population = actual_class_population.iloc[:3]
        print(expected_class_population)
        print(actual_class_population)
        # todo check why it is not returning equal
        print(expected_class_population.equals(actual_class_population))
        self.assertEqual(expected_class_population.equals(actual_class_population), True,
                         'Expected != Actual')


if __name__ == '__main__':
    unittest.main()
