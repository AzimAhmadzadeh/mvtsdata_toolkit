import unittest
import os

import CONSTANTS as CONST
from configs.config_reader import ConfigReader


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_non_existing_config_file(self):
        """ Checks if the method raises an exception if the config file does not exist."""
        path_to_config = os.path.join(CONST.ROOT, 'tests/configs/non_existing_file.yml')
        cr = ConfigReader(path_to_config)
        with self.assertRaises(FileNotFoundError):
            cr.read()

    def test_non_yml_config_filename(self):
        """ Checks if the method raises an exception if the config file exists but it is NOT a yml
        file."""
        path_to_config = os.path.join(CONST.ROOT, 'tests/configs/non_yml_config_file.txt')
        cr = ConfigReader(path_to_config)
        with self.assertRaises(FileNotFoundError):
            cr.read()

    def test_config_with_missing_keys(self):
        """ Checks if the method raises an exception if the config file misses one of the keys in
        its content."""
        path_to_config = os.path.join(CONST.ROOT, 'tests/configs/configs_with_missing_key.yml')
        cr = ConfigReader(path_to_config)
        with self.assertRaises(AssertionError):
            cr.read()

    def test_config_with_extra_keys(self):
        """ Checks if the method raises an exception if the config file has an extra key in its
        content."""
        path_to_config = os.path.join(CONST.ROOT, 'tests/configs/configs_with_extra_key.yml')
        cr = ConfigReader(path_to_config)
        with self.assertRaises(AssertionError):
            cr.read()

    def test_config_with_missing_values(self):
        """ xx """
        path_to_config = os.path.join(CONST.ROOT,
                                      'tests/configs/configs_with_no_PATH_TO_EXTRACTED_FEATURES.yml')
        cr = ConfigReader(path_to_config)
        # with self.assertRaises(AssertionError):
        configs = cr.read()
        print(configs['PATH_TO_EXTRACTED_FEATURES'])


if __name__ == '__main__':
    unittest.main()
