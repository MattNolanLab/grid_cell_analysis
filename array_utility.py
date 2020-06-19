import numpy as np


# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(array_to_shift, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), array_to_shift[:-n]))
    else:
        return np.concatenate((array_to_shift[-n:], np.full(-n, np.nan)))


'''
Shifts 2d array along given axis.

array_to_shift : 2d array that is to be shifted
n : array will be shifted by n places
axis : shift along this axis (should be 0 or 1)
'''


def shift_2d(array_to_shift, n, axis):
    shifted_array = np.zeros_like(array_to_shift)
    if axis == 0:  # shift along x axis
        if n == 0:
            return array_to_shift
        if n > 0:
            shifted_array[:, :n] = 0
            shifted_array[:, n:] = array_to_shift[:, :-n]
        else:
            shifted_array[:, n:] = 0
            shifted_array[:, :n] = array_to_shift[:, -n:]

    if axis == 1:  # shift along y axis
        if n == 0:
            return array_to_shift
        elif n > 0:
            shifted_array[-n:, :] = 0
            shifted_array[:-n, :] = array_to_shift[n:, :]
        else:
            shifted_array[:-n, :] = 0
            shifted_array[-n:, :] = array_to_shift[:n, :]
    return shifted_array


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
         # linear interpolation of NaNs
         nans, x= nan_helper(y)
         y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def remove_nans_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2


def remove_nans_and_inf_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]

    not_nans_in_array1 = ~np.isinf(array1)
    not_nans_in_array2 = ~np.isinf(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = -2
    axis = 1

    desired_result = np.array([[np.nan, np.nan, 1, 1], [np.nan, np.nan, 2, 9], [np.nan, np.nan, 3, 3], [np.nan, np.nan, 4, 4], [np.nan, np.nan, 5, 5], [np.nan, np.nan, 6, 6]])
    result = shift_2d(array_to_shift, n, axis)

    array_to_shift2 = np.array([[[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3]], [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]])


if __name__ == '__main__':
    main()

