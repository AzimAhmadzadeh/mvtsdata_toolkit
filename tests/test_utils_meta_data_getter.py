import unittest
import CONSTANTS as CONST
import os
import pandas as pd
import utils.meta_data_getter as mdg

# todo test exception as per python
class TestMetaDataGetter(unittest.TestCase):

    good_file_name = None
    bad_file_name = None

    @classmethod
    def setUpClass(cls) -> None:
        path_to_root = CONST.IN_PATH_TO_MVTS

        cls.good_file_name = 'lab[B]1.0@1053_id[345]_st[2011-01-24T03:24:00]_et[2011-01-24T11:12:00].csv'
        cls.bad_file_name = 'lab[[B]]1.0@1053_id[345[]]_st[2011-01-24T[03:24:00]]_et[2011-01-24T[11:12:00]].csv'

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    # good_file_name --> Filename is well formatted doesn't contain any extra braces in substrings
    # Below methods test over filename which doesn't contain any extra paired braces

    def test_05_extract_id(self):
        expected_id = '345'
        actual_id = mdg.extract_id(TestMetaDataGetter.good_file_name, 'id')
        self.assertEqual(expected_id, actual_id, 'Expected != Actual')

    def test_06_extract_start_time(self):
        expected_time = '2011-01-24T03:24:00'
        actual_time = mdg.extract_start_time(TestMetaDataGetter.good_file_name, 'st')
        self.assertEqual(expected_time, actual_time, 'Expected != Actual')

    def test_07_extract_end_time(self):
        expected_time = '2011-01-24T11:12:00'
        actual_time = mdg.extract_start_time(TestMetaDataGetter.good_file_name, 'et')
        self.assertEqual(expected_time, actual_time, 'Expected != Actual')

    def test_08_extract_class_label(self):
        flare_class = mdg.extract_class_label(TestMetaDataGetter.good_file_name, 'lab')
        self.assertEqual(flare_class, 'B', 'Extracted class labels from file name is incorrect.')

    # bad_file_name --> Filename has extra [] but no unpaired brace ;
    # Below methods test over filename which has extra pair of braces in substrings to be extracted

    def test_08_extract_id(self):
        expected_id = '345[]'
        actual_id = mdg.extract_id(TestMetaDataGetter.bad_file_name, 'id')
        print(actual_id)
        self.assertEqual(expected_id, actual_id, 'Expected != Actual')

    def test_09_extract_start_time(self):
        expected_time = '2011-01-24T[03:24:00]'
        actual_time = mdg.extract_start_time(TestMetaDataGetter.bad_file_name, 'st')
        self.assertEqual(expected_time, actual_time, 'Expected != Actual')

    def test_10_extract_end_time(self):
        expected_time = '2011-01-24T[11:12:00]'
        actual_time = mdg.extract_start_time(TestMetaDataGetter.bad_file_name, 'et')
        self.assertEqual(expected_time, actual_time, 'Expected != Actual')

    def test_11_extract_class_label(self):
        flare_class = mdg.extract_class_label(TestMetaDataGetter.bad_file_name, 'lab')
        self.assertEqual(flare_class, '[B]', 'Extracted class labels from file name is incorrect.')

    # Test of exception --> by providing wrong starting phrase
    def test_12_extract_class_label(self):
        flare_class = mdg.extract_class_label(TestMetaDataGetter.bad_file_name, 'lbb')
        self.assertEqual(flare_class, '[B]', 'Extracted class labels from file name is incorrect.')


if __name__ == '__main__':
    unittest.main()
