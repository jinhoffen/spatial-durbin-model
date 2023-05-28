"""This module computes the negative sample-average of the log likelihood function for static and dynamic spatial GAS models."""

import numpy as np

from sdm.specification import Hyperparameter

from sdm.utils import rebuild_parameters
from sdm import filter

def static_heterogenous(
    params: np.ndarray,
    specification: Hyperparameter,
    mY: np.ndarray,
    mmX: np.ndarray,
    mmW: np.ndarray,
    iP: int
) -> float:
    """
    Calculate likelihood function for static heterogenous model

    Parameters
    ----------
    params: np.ndarray
        Flattened parameter vector. Dimensions: iP + iK + 1. Order: vMu, vBeta, dSigma2.
    specification: Hyperparameter

    mY: np.ndarray
        Observation data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iT x iN x iK.
    mmW:
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.

    Returns
    -------
    Negative of the average log likelihood function value

    Notes
    -----
    Return the negative of the average log-likelihood because we apply a minimization algorithm.
    We compute the average log-likelihood to stabilize the algorithm's behavior.
    """
    # parse values from parameter array
    iK = mmX.shape[2]

    # obtain parameters
    vOmega, vBeta, dSigma2 = rebuild_parameters(params, iP, iK, 1)

    params = {
        "vOmega": vOmega,
        "vBeta": vBeta,
        "dSigma2": dSigma2
    }

    _, avg_log_lik = filter.static_heterogenous(mY, mmX, mmW, params, specification)

    return -avg_log_lik

def dynamic_heterogenous(
    params: np.ndarray,
    specification: Hyperparameter,
    mY: np.ndarray,
    mmX: np.ndarray,
    mmW: np.ndarray,
    iP: int
):
    """
    Calculate Likelihood Function for dynamic spatial gas model

    Parameters
    ----------
    params: np.ndarray
        Flattened parameter vector. Order: vMu_init, vBeta_init, dSigma2_init
    specification: Hyperparameter
        Model hyperparameters.
    mY: np.ndarray
        Observation data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iT x iN x iK.
    mmW:
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.

    Returns
    -------
    Negative of the average log likelihood function value

    Notes
    -----
    Return the negative of the average log-likelihood because we apply a minimization algorithm.
    We compute the average log-likelihood to stabilize the algorithm's behavior.

    Todo
    ----
    Change initialization to np.linalg.inv(np.identity()-B).dot(omega)
    """
    # parse values from parameter array
    iK = mmX.shape[2]

    # obtain parameter
    vOmega, vA, vB, vBeta, dSigma2 = rebuild_parameters(params, iP, iP, iP, iK, 1)

    params = {
        "vOmega": vOmega,
        "vA": vA,
        "vB": vB,
        "vBeta": vBeta,
        "dSigma2": dSigma2
    }

    # compute log likelihood function value
    _, _, _, avg_log_lik  = filter.dynamic_heterogenous(mY, mmX, mmW, params, specification)

    return -avg_log_lik