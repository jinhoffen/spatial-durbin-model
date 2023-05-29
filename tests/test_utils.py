import numpy as np

from sdm import utils

def test_flatten_parameters():
    """Test flattening of parameters"""
    # test data
    arr_lengths = (2, 6, 2, 1)

    params = []
    start = 0
    for length in arr_lengths:
        params.append(np.arange(start, start+length))
        start += length

    # expected data
    expected = np.arange(np.sum(arr_lengths))

    # test
    res = utils.flatten_parameters(*params)

    assert np.array_equal(res, expected)

def test_rebuild_parameters():
    """Test rebuilding of parameters"""
    params = np.arange(10)
    lengths = (3, 5, 1)
    arrs = utils.rebuild_parameters(params, *lengths)

    start = 0
    for arr, length in zip(arrs, lengths):
        assert arr.shape[0] == length
        assert np.array_equal(np.arange(start, start+length), arr)
        start += length