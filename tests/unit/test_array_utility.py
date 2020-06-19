import numpy as np
import sys
sys.path.append("..")
import array_utility


def test_shift():
    array_to_shift = np.array([0, 1, 2, 3])
    n = 2
    desired_result = np.array([0, 0, 0, 1])
    result = array_utility.shift(array_to_shift, n)

    n = -2
    desired_result = np.array([2, 3, 0, 0])
    result = array_utility.shift(array_to_shift, n)


def test_shift_2d():
    # shift right along x
    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = 2
    axis = 0

    desired_result = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3], [0, 0, 4, 4], [0, 0, 5, 5], [0, 0, 6, 6]])
    result = array_utility.shift_2d(array_to_shift, n, axis)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    # shift left along x
    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = -2
    axis = 0

    desired_result = np.array([[1, 1, 0, 0], [2, 9, 0, 0], [3, 3, 0, 0], [4, 4, 0, 0], [5, 5, 0, 0], [6, 6, 0, 0]])
    result = array_utility.shift_2d(array_to_shift, n, axis)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    # shift up
    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = 2
    axis = 1

    desired_result = np.array([[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [0, 0, 0, 0], [0, 0, 0, 0]])
    result = array_utility.shift_2d(array_to_shift, n, axis)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    # shift down
    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = -2
    axis = 1

    desired_result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4]])
    result = array_utility.shift_2d(array_to_shift, n, axis)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)