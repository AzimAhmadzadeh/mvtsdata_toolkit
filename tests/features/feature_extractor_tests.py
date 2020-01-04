import unittest
import os
import CONSTANTS as CONST
from features.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_1X_constructor(self):
        """ Constructor throws an exception if the config file does not exist."""
        path_to_config = '/invalid_path.yml'
        with self.assertRaises(Exception):
            FeatureExtractor(path_to_config)

    def test_2X_constructor(self):
        """ Constructor throws an exception if the config file is NOT a yml file."""
        path_to_config = os.path.join(CONST.ROOT, 'CONSTANTS.py')
        with self.assertRaises(Exception):
            FeatureExtractor(path_to_config)

    def test_params_in_do_extraction_X1(self):
        """
        Checks if the method raises proper exception when:
            - `MVTS_PARAMETERS` is empty in the config file, and
            - none of the args `params_name` and `params_index` are given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/configs_with_no_MVTS_PARAMETERS.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=[], params_index=[], features_index=[0, 1])

    def test_params_in_do_extraction_X2(self):
        """
        Checks if the method raises proper exception when:
            - `MVTS_PARAMETERS` is empty in the config file, and
            - only the args `params_index` is given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/configs_with_no_MVTS_PARAMETERS.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=[], params_index=[0, 1], features_index=[0, 1])

    def test_params_in_do_extraction_X3(self):
        """
        Checks if the method raises proper exception when:
            - `MVTS_PARAMETERS` is empty in the config file, and
            - both of the args `params_name` and `params_index` are given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/feature_extraction_configs.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=['TOTUSJH', 'TOTBSQ'], params_index=[0, 1],
                             features_index=[0, 1])

    def test_features_in_do_extraction_X1(self):
        """
        Checks if the method raises proper exception when:
            - `STATISTICAL_FEATURES` is empty in the config file, and
            - none of the args `features_name` and `features_index` are given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/configs_with_no_STATISTICAL_FEATURES.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=['TOTUSJH', 'TOTBSQ'], features_name=[],
                             features_index=[])

    def test_features_in_do_extraction_X2(self):
        """
        Checks if the method raises proper exception when:
            - `STATISTICAL_FEATURES` is empty in the config file, and
            - only the args `features_index` is given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/configs_with_no_STATISTICAL_FEATURES.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=['TOTUSJH', 'TOTBSQ'], features_name=[],
                             features_index=[0, 1])

    def test_features_in_do_extraction_X3(self):
        """
        Checks if the method raises proper exception when:
            - `STATISTICAL_FEATURES` is empty in the config file, and
            - both the args `features_name` and `features_index` are given.
        """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/feature_extraction_configs.yml')
        fe = FeatureExtractor(path_to_config)
        with self.assertRaises(ValueError):
            fe.do_extraction(params_name=['TOTUSJH', 'TOTBSQ'],
                             features_name=['get_min', 'get_max'],
                             features_index=[0, 1])

    def test_store_extracted_features_X1(self):
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/feature_extraction_configs.yml')
        output_filename = 'extracted_features_TEST_OUTPUT.csv'
        output_path = os.path.join(CONST.ROOT,
                                      'tests/test_dataset/extracted_features/', output_filename)

        fe = FeatureExtractor(path_to_config)
        fe.do_extraction(params_name=['TOTUSJH', 'TOTBSQ'], features_index=[0, 1])
        fe.store_extracted_features(output_filename, verbose=False)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
