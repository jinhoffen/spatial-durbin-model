"""This module tests the expected behavior of basic numpy operations."""

import numpy as np

def test_transpose():
    """test transpose of 3d array"""
    iP = 2
    iN = 5
    mmW = np.random.rand(iP, iN, iN)

    mmW_transposed = mmW.transpose(0, 2, 1)

    for i in range(iP):
        expected = mmW[i].T
        np.testing.assert_array_equal(expected, mmW_transposed[i])

def test_dot_product():
    """test dot product of 1d and 3d array"""
    iN = 5
    iP = 2
    vY = np.random.rand(iN)
    mmW = np.random.rand(iP, iN, iN)

    dot_product = vY @ mmW

    for i in range(iP):
        expected = vY.dot(mmW[i])
        np.testing.assert_array_equal(expected, dot_product[i])


def test_elementwise_product():
    """test element wise product of 1d and 3d array"""
    iN = 5
    iP = 2
    vRho = np.random.rand(iP)
    mmW = np.random.rand(iP, iN, iN)

    elementwise_product = np.multiply(vRho[:,None,None], mmW)

    for i in range(iP):
        expected = vRho[i] * mmW[i]
        np.testing.assert_array_equal(expected, elementwise_product[i])

def test_sum():
    """test sum across first dimensions of 3d array"""
    iN = 5
    iP = 2

    mmW = np.random.rand(iP, iN, iN)
    arr_sum = np.sum(mmW, axis=0)
    expected = np.sum([mmW[i] for i in range(iP)], axis=0)

    np.testing.assert_array_equal(expected, arr_sum)

def test_trace():
    """test trace of 3d array"""
    iN = 5
    iP = 2

    mmW = np.random.rand(iP, iN, iN)

    trace = mmW.trace(axis1=1, axis2=2)

    for i in range(iP):
        expected = mmW[i].trace()
        np.testing.assert_array_equal(expected, trace[i])