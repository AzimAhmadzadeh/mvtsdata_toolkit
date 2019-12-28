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

    def test(self):
        path_to_config = os.path.join(CONST.ROOT, 'CONSTANTS.py')
        FeatureExtractor(path_to_config)
        # print(os.path.isfile(path_to_config))
        # print(os.path.exists(path_to_config))


if __name__ == '__main__':
    unittest.main()
