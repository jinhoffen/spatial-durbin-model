"""This module provides utility functions."""

import numpy as np
import pandas as pd

from sdm.specification import Hyperparameter

def score(
    vYt: np.ndarray,
    mXt: np.ndarray,
    vFt: np.ndarray,
    mmW: np.ndarray,
    specification: Hyperparameter,
    vBeta: np.ndarray,
    mSigma: np.ndarray,
    vRho: np.ndarray
) -> np.ndarray:
    """Calculate score"""
    # parse parameters
    iN = vYt.shape[0]
    iP = vRho.shape[0]

    # compute Z_t
    # mContagion = np.sum(np.multiply(vRho[:,None,None], mmW), axis=0)
    R = np.kron(np.diag(vRho), np.identity(iN//iP))

    Zt = np.linalg.inv(np.identity(iN) - (R @ mmW))
    residual = vYt - (R @ mmW).dot(vYt) - mXt.dot(vBeta)

    # compute score
    loglik_foc = ((vYt @ mmW.T) @ np.linalg.inv(mSigma)).dot(residual) - (Zt @ mmW).trace()
    score = np.multiply(loglik_foc, specification.transformation_foc(vFt))

    return score

def scaled_score(
    score: np.ndarray,
    mScaling: np.ndarray
) -> np.ndarray:
    """Calculate scaled score"""
    return mScaling.dot(score)

def flatten_parameters(*params: np.ndarray) -> np.ndarray:
    """Returns flattened parameter vector"""
    return np.concatenate(list(params))

def rebuild_parameters(
    params: np.ndarray,
    *breakpoints: int
) -> list:
    """Rebuild parameter arrays from concatenated parameters"""
    parameter_list = []
    cutoff = 0

    for breakpoint in breakpoints:
        parameter_list.append(params[cutoff:cutoff+breakpoint])
        cutoff += breakpoint

    return parameter_list

def multivariate_normal_native(
    size: int,
    mean: np.ndarray,
    cov: np.ndarray
):
    """
    Draw from multivariate normal distribution

    Notes
    -----
    np.random.multivariate_normal is not supported by numba
    """
    x = np.random.standard_normal(size)
    y = np.reshape(x, (-1, mean.shape[0]))

    cov = cov.astype(np.double)
    l = np.linalg.cholesky(cov)
    z = mean + np.dot(y, l)

    return np.reshape(z, size)

def compare(*param_sets):
    """"""
    param_labels = ["vOmega", "vA", "vB", "vBeta", "dSigma2"]

    result = []

    for param_label in param_labels:
        d = dict(label = param_label)
        for set_label, set_params in param_sets:
            d[set_label] = set_params[param_label]

        result.append(pd.DataFrame(d))

    return pd.concat(result, axis=0)