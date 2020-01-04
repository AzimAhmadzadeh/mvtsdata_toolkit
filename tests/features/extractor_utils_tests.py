import pandas as pd
import numpy as np
import unittest
import features.extractor_utils as ex_util
from pandas.util.testing import assert_frame_equal


class TestExtractorUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_methods_for_names(self):
        """ Tests whether callable methods can be retrieved based on method names."""
        method_names = ['get_min', 'get_max']
        callable_methods = ex_util.get_methods_for_names(method_names)
        is_callable = [callable(m) for m in callable_methods]
        self.assertEqual(is_callable, [True] * len(method_names))

    def test_flatten_to_row_df1(self):
        """ Input is a 2X3 dataframe, and the expected output is a 1X6 dataframe. """
        df_input = pd.DataFrame({'param1': [1, 4],
                                 'param2': [2, 5],
                                 'param3': [3, 6]}, index=['row1', 'row2'])
        df_expected = pd.DataFrame([[1, 2, 3, 4, 5, 6]], index=[0],
                                   columns=['row1_param1', 'row1_param2', 'row1_param3',
                                            'row2_param1', 'row2_param2', 'row2_param3'])
        df_flattened = ex_util.flatten_to_row_df(df_input)
        assert_frame_equal(df_expected, df_flattened, check_dtype=False)

    def test_flatten_to_row_df2(self):
        """ Input is a 3X1 dataframe, and the expected output is a 1X3 dataframe. """

        df_input = pd.DataFrame([1, 2, 3], index=['row1', 'row2', 'row3'], columns=['param1'])
        df_expected = pd.DataFrame([[1, 2, 3]], index=[0],
                                   columns=['row1_param1', 'row2_param1', 'row3_param1'])
        df_flattened = ex_util.flatten_to_row_df(df_input)
        assert_frame_equal(df_expected, df_flattened, check_dtype=False)

    def test_split(self):
        """ Split a list of 200 elements into 3 partitions of approximately equal size."""
        l = list(np.arange(start=1, stop=201, step=1))
        s = ex_util.split(l, 3)
        split_sizes = [len(s[0]), len(s[1]), len(s[2])]
        expected_sizes = [67, 67, 66]  # approximately equal sizes
        self.assertListEqual(split_sizes, expected_sizes)
        total_expected = np.sum(expected_sizes)
        total = len(l)
        self.assertEqual(total, total_expected)


if __name__ == '__main__':
    unittest.main()
