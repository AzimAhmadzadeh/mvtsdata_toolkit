import unittest
from data_analysis import mvts_data_analysis
import numpy as np


class TestMVTSDataAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        f_path = 'tests/test_dataset/mvts'
        cls.mvts = mvts_data_analysis.MVTSDataAnalysis(f_path)
        cls.mvts.compute_summary()

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_missing_values(self):
        expected_class_population = np.array([3, 1, 1], dtype=int)

        actual_class_population = self.mvts.get_missing_values()
        actual_class_population = actual_class_population.iloc[:3]
        actual_class_population = actual_class_population['Null-Count'].values

        np.testing.assert_equal(actual_class_population, expected_class_population)


if __name__ == '__main__':
    unittest.main()
