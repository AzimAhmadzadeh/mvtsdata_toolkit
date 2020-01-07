import unittest
import os
import pandas as pd

from sampling.sampler import Sampler
import CONSTANTS as CONST


class TestSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path_to_mvts = os.path.join(CONST.ROOT,
                                    'tests/test_dataset/extracted_features'
                                    '/extracted_features_TESTS_SAMPLER.csv')
        cls.mvts_df = pd.read_csv(path_to_mvts, sep='\t')

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_class_labels(self):
        """ Tests if sampler returns the class labels correctly."""
        sampler = Sampler(self.mvts_df, 'lab')
        expected_labels = {'X', 'M', 'C', 'NF'}
        actual_labels = set(sampler.class_labels)
        self.assertSetEqual(actual_labels, expected_labels)

    def test_class_populations(self):
        """ Tests if sampler returns class populations correctly."""
        sampler = Sampler(self.mvts_df, 'lab')
        expected_dict = {'NF': 36, 'M': 4, 'X': 2, 'C': 8}
        actual_dict = sampler.class_population
        self.assertDictEqual(actual_dict, expected_dict)

    def test_class_ratios(self):
        """ Tests if sampler returns class ratios correctly."""
        sampler = Sampler(self.mvts_df, 'lab')
        expected_dict = {'NF': 0.72, 'M': 0.08, 'X': 0.04, 'C': 0.16}
        actual_dict = sampler.class_ratios
        self.assertDictEqual(actual_dict, expected_dict)

    def test_get_original_populations(self):
        sampler = Sampler(self.mvts_df, 'lab')
        sampler.ori


if __name__ == '__main__':
    unittest.main()
