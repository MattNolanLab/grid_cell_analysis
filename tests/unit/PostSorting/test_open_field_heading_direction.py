import numpy as np
import PostSorting.open_field_heading_direction


def test_calculate_heading_direction():
    x = [0, 1, 2, 2, 1]
    y = [0, 1, 1, 0, 1]

    desired_result = [225, 225, 180, 90, 315]
    result = PostSorting.open_field_heading_direction.calculate_heading_direction(x, y, pad_first_value=True)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)


    desired_result = [45 + 180, 0 + 180, -90 + 180, 135 + 180]
    result = PostSorting.open_field_heading_direction.calculate_heading_direction(x, y, pad_first_value=False)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)
