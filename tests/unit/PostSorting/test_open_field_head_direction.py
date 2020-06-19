import numpy as np
import PostSorting.open_field_head_direction


def test_get_rolling_sum():

    array_in = [1, 2, 3, 4, 5, 6]
    window = 3

    desired_result = [9, 6, 9, 12, 15, 12]
    result = PostSorting.open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5]
    window = 3

    desired_result = [12, 12, 17, 24, 20, 15, 9, 11]
    result = PostSorting.open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5, 4]
    window = 3

    desired_result = [11, 12, 17, 24, 20, 15, 9, 12, 12]
    result = PostSorting.open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)


def test_get_rayleighscore_for_cluster():
    hd_hist = np.ones(360)
    expected_result = 1  # uniform distribution
    result = PostSorting.open_field_head_direction.get_rayleigh_score_for_cluster(hd_hist)
    assert np.allclose(result, expected_result, rtol=1e-05, atol=1e-08)

    hd_hist = np.ones(20) * 300
    expected_result = 1  # uniform distribution (different array shape)
    result = PostSorting.open_field_head_direction.get_rayleigh_score_for_cluster(hd_hist)
    assert np.allclose(result, expected_result, rtol=1e-05, atol=1e-08)



