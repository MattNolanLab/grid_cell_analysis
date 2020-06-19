import numpy as np
import PostSorting.open_field_light_data
import pandas as pd
from pandas.util.testing import assert_frame_equal


def test_make_opto_data_frame():

    # pulses equally spaced and same length
    array_in = ([[1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21]])
    desired_df = pd.DataFrame()
    desired_df['opto_start_times'] = [1, 9, 17]
    desired_df['opto_end_times'] = [5, 13, 21]

    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert assert_frame_equal(desired_df, result_df, check_dtype=False) is None


    # lengths of pulses are different
    array_in = ([[1, 2, 3, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27]])
    desired_df = pd.DataFrame()
    desired_df['opto_start_times'] = [1, 10, 21]
    desired_df['opto_end_times'] = [3, 14, 27]

    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert assert_frame_equal(desired_df, result_df, check_dtype=False) is None

    # spacings between pulses are different
    array_in = ([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27]])
    desired_df = pd.DataFrame()
    desired_df['opto_start_times'] = [1, 10, 23]
    desired_df['opto_end_times'] = [5, 14, 27]

    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert assert_frame_equal(desired_df, result_df, check_dtype=False) is None

    # lengths and spacing between pulses are different
    array_in = ([[1, 2, 3, 10, 11, 12, 13, 14, 26, 27, 28, 29, 30, 31, 32]])
    desired_df['opto_start_times'] = [1, 10, 26]
    desired_df['opto_end_times'] = [3, 14, 32]

    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert assert_frame_equal(desired_df, result_df, check_dtype=False) is None

    # pulse start != 1
    array_in = ([[10, 11, 12, 13, 14, 26, 27, 28, 29, 30, 42, 43, 44, 45, 46]])
    desired_df['opto_start_times'] = [10, 26, 42]
    desired_df['opto_end_times'] = [14, 30, 46]

    result_df = PostSorting.open_field_light_data.make_opto_data_frame(array_in)
    assert assert_frame_equal(desired_df, result_df, check_dtype=False) is None


def main():
    test_make_opto_data_frame()


if __name__ == '__main__':
    main()
