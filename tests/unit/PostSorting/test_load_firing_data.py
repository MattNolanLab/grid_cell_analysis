import numpy as np
import PostSorting.load_firing_data


def test_correct_detected_ch_for_dead_channels():
    dead_channels = [1, 2]
    dead_channels = list(map(int, dead_channels))
    primary_channels = np.array([1, 1, 3, 6, 7, 9, 11, 3])

    desired_result = [3, 3, 5, 8, 9, 11, 13, 5]
    result = PostSorting.load_firing_data.correct_detected_ch_for_dead_channels(dead_channels, primary_channels)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)
