import unittest
import os
import pandas as pd
import numpy as np
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

    def test_original_class_populations(self):
        """ Tests if sampler returns class populations correctly."""
        sampler = Sampler(self.mvts_df, 'lab')
        expected_dict = {'NF': 36, 'M': 4, 'X': 2, 'C': 8}
        actual_dict = sampler.original_class_populations
        self.assertDictEqual(actual_dict, expected_dict)

    def test_original_class_ratios(self):
        """ Tests if sampler returns class ratios correctly."""
        sampler = Sampler(self.mvts_df, 'lab')
        expected_dict = {'NF': 0.72, 'M': 0.08, 'X': 0.04, 'C': 0.16}
        actual_dict = sampler.original_class_ratios
        self.assertDictEqual(actual_dict, expected_dict)

    def test_sampled_class_populations_x1(self):
        """ Tests if sampler samples the right populations when desired populations are given."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = {'NF': 10, 'M': 10, 'X': 10, 'C': 10}     # ---> given are populations
        desrired_ratios = None
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = desired_populations
        self.assertDictEqual(sampler.sampled_class_populations, expected_populations)

    def test_sampled_class_populations_x2(self):
        """ Tests if sampler can handle 0 as desired populations."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = {'NF': 36, 'M': 0, 'X': 0, 'C': 8}     # ---> 0 populations
        desrired_ratios = None
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = desired_populations
        self.assertDictEqual(sampler.sampled_class_populations, expected_populations)

    def test_sampled_class_populations_x3(self):
        """ Tests if sampler can handle -1 as desired populations."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = {'NF': -1, 'M': 0, 'X': 0, 'C': -1}     # ---> -1 populations
        desrired_ratios = None
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = {'NF': 36, 'M': 0, 'X': 0, 'C': 8}
        self.assertDictEqual(sampler.sampled_class_populations, expected_populations)

    def test_sampled_class_populations_x4(self):
        """ Tests if sampler samples the right populations when desired ratios are given."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': 0.10, 'M': 0.10, 'X': 0.10, 'C': 0.10}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = {'NF': 5, 'M': 5, 'X': 5, 'C': 5}   # 5 = 0.10 X total population
        self.assertDictEqual(sampler.sampled_class_populations, expected_populations)

    def test_sampled_class_populations_x5(self):
        """ Tests if sampler can handle 0 as desired ratios, in terms of sampled populations."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': 0.50, 'M': 0.0, 'X': 0.0, 'C': 0.50}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = {'NF': 25, 'M': 0.0, 'X': 0.0, 'C': 25}  # 25 = 0.5 X total
        # population
        self.assertDictEqual(sampler.sampled_class_populations, expected_populations)

    def test_sampled_class_populations_x6(self):
        """ Tests if sampler can handle -1 as desired ratios, in terms of sampled populations."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': -1, 'M': 0.0, 'X': 0.0, 'C': -1}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_populations = {'NF': 36, 'M': 0, 'X': 0, 'C': 8}  # [36, 0, 0, 8]
        np.testing.assert_array_almost_equal(list(sampler.sampled_class_populations.values()),
                                             list(expected_populations.values()),
                                             decimal=2)

    def test_sampled_class_ratios_x1(self):
        """ Tests if sampler samples the right ratios when desired ratios are given."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': 0.10, 'M': 0.10, 'X': 0.10, 'C': 0.10}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_ratios = {'NF': 0.25, 'M': 0.25, 'X': 0.25, 'C': 0.25}  # 25% of new population
        self.assertDictEqual(sampler.sampled_class_ratios, expected_ratios)

    def test_sampled_class_ratios_x2(self):
        """ Tests if sampler can handle 0 as desired ratios, in terms of sampled ratios."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': 0.5, 'M': 0.0, 'X': 0.0, 'C': 0.50}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_ratios = {'NF': 0.5, 'M': 0.0, 'X': 0.0, 'C': 0.50}  # 50% of new population
        self.assertDictEqual(sampler.sampled_class_ratios, expected_ratios)

    def test_sampled_class_ratios_x3(self):
        """ Tests if sampler can handle -1 as desired ratios, in terms of sampled ratios."""
        sampler = Sampler(self.mvts_df, 'lab')
        desired_populations = None
        desrired_ratios = {'NF': -1, 'M': 0.0, 'X': 0.0, 'C': -1}  # ---> given are ratios
        sampler.sample(desired_populations=desired_populations,
                       desired_ratios=desrired_ratios)
        expected_ratios = {'NF': 0.82, 'M': 0.0, 'X': 0.0, 'C': 0.18}  # [36/44, 0, 0, 8/44]
        np.testing.assert_array_almost_equal(list(sampler.sampled_class_ratios.values()),
                                             list(expected_ratios.values()),
                                             decimal=2)

    def test_undersample_x1(self):
        """ Tests if undersampler samples correctly with based_minority set to 'NF'."""
        sampler = Sampler(self.mvts_df, 'lab')
        minority_labels = ['NF', 'C']
        majority_labels = ['X', 'M']
        base_minority = 'NF'                    # ---> base class is set to 'NF'
        sampler.undersample(minority_labels=minority_labels,
                            majority_labels=majority_labels,
                            base_minority=base_minority)

        expected_populations = {'NF': 36, 'M': 36, 'X': 36, 'C': 36}     # |NF|=36 in original mvts
        expected_ratios = {'NF': 0.25, 'M': 0.25, 'X': 0.25, 'C': 0.25}  # 0.25 = 36 / (4 X 36)

        self.assertDictEqual(expected_populations, sampler.sampled_class_populations)
        self.assertDictEqual(expected_ratios, sampler.sampled_class_ratios)

    def test_undersample_x2(self):
        """ Tests if undersampler samples correctly with based_minority set to 'C'."""
        sampler = Sampler(self.mvts_df, 'lab')
        minority_labels = ['NF', 'C']
        majority_labels = ['X', 'M']
        base_minority = 'C'                    # ---> base class is set to 'C'
        sampler.undersample(minority_labels=minority_labels,
                            majority_labels=majority_labels,
                            base_minority=base_minority)

        expected_populations = {'NF': 8, 'M': 8, 'X': 8, 'C': 8}     # |C|=8 in original mvts
        self.assertDictEqual(expected_populations, sampler.sampled_class_populations)

    def test_undersample_x3(self):
        """ Tests if undersampler samples correctly with based_minority set to 'C'."""
        sampler = Sampler(self.mvts_df, 'lab')
        minority_labels = ['NF', 'C']
        majority_labels = ['X', 'M']
        base_minority = 'C'                    # ---> base class is set to 'C'
        sampler.undersample(minority_labels=minority_labels,
                            majority_labels=majority_labels,
                            base_minority=base_minority)

        expected_ratios = {'NF': 0.25, 'M': 0.25, 'X': 0.25, 'C': 0.25}  # 0.25 = 8 / (4 X 8)
        self.assertDictEqual(expected_ratios, sampler.sampled_class_ratios)

    def test_oversample_x1(self):
        """ Tests if oversampler samples correctly with based_majority set to 'M'."""
        sampler = Sampler(self.mvts_df, 'lab')
        minority_labels = ['NF', 'C']
        majority_labels = ['X', 'M']
        base_majority = 'M'                    # ---> base class is set to 'M'
        sampler.oversample(minority_labels=minority_labels,
                           majority_labels=majority_labels,
                           base_majority=base_majority)

        expected_populations = {'NF': 4, 'M': 4, 'X': 4, 'C': 4}  # |M|=4 in original mvts
        self.assertDictEqual(expected_populations, sampler.sampled_class_populations)

        expected_ratios = {'NF': 0.25, 'M': 0.25, 'X': 0.25, 'C': 0.25}  # 0.25 = 4 / (4 X 4)
        self.assertDictEqual(expected_ratios, sampler.sampled_class_ratios)


if __name__ == '__main__':
    unittest.main()
