"""This module computes the filter."""

import numpy as np

from sdm.utils import score, scaled_score
from sdm.model import Hyperparameter

def static_heterogenous(
    mY: np.ndarray,
    mmX: np.ndarray,
    mmW: np.ndarray,
    params: dict,
    specification: Hyperparameter
):
    """
    Predictive filter for the static heterogenous model

    Parameters
    ----------
    mY: np.ndarray
        Observation data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iP x iN x iK.
    mmW: np.ndarray
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.
    params: dict
        Parameters of the time-varying parameter vector. Dimension: iP.
    specification: Hyperparameter
        Model hyperparameters.

    Return
    ------
    mf: np.ndarray
        Time-varying parameter vector
    L: float
        Average log-likelihood function
    """
    iT, iN, _ = mmW.shape

    vOmega = params["vOmega"]
    vBeta = params["vBeta"]
    mSigma = np.identity(iN) * params["dSigma2"]

    iP = len(params["vOmega"])

    # generate filter
    mf = np.tile(vOmega, (iT+1, 1))

    # generate log likelihood
    vloglik = np.zeros(iT)
    vRho = specification.transformation(mf[0])
    for t in range(iT):
        R = np.kron(np.diag(vRho), np.identity(iN//iP))
        residual = mY[t] - (R @ mmW[t]).dot(mY[t]) - mmX[t].dot(vBeta)
        vloglik[t] = np.log(np.linalg.det(np.identity(iN) - (R @ mmW[t]))) - iN * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(mSigma)) - 0.5 * residual.dot(np.linalg.inv(mSigma)).dot(residual)

    avg_log_lik = 1/(iT * iN) * np.sum(vloglik)

    return mf, avg_log_lik

def dynamic_heterogenous(
    mY: np.ndarray,
    mmX: np.ndarray,
    mmW: np.ndarray,
    params: dict,
    specification: Hyperparameter
):
    """
    Predictive filter for the dynamic heterogenous model

    Parameters
    ----------
    mY: np.ndarray
        Observation data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iP x iN x iK.
    mmW: np.ndarray
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.
    params: dict
        Parameters of the time-varying parameter vector. Dimension: iP.
    specification: Hyperparameter
        Model hyperparameters.

    Return
    ------
    mf: np.ndarray
        Time-varying parameter vector
    L: float
        Average log-likelihood function
    """
    iT, iN, _ = mmW.shape

    vOmega = params["vOmega"]
    mA = np.diag(params["vA"])
    mB = np.diag(params["vB"])
    vBeta = params["vBeta"]
    mSigma = np.identity(iN) * params["dSigma2"]

    iP = len(params["vOmega"])

    # initialize time varying parameter
    mf = np.zeros((iT+1, iP))
    mf[0] = np.linalg.inv(np.identity(iP) - mB).dot(vOmega)

    # initialize score and innovations
    mScores = np.zeros((iT, iP))
    mInnovations = np.zeros((iT, iP))

    # initialize log likelihood
    vloglik = np.zeros(iT)

    # generate log likelihood
    for t in range(iT):
        vRho = specification.transformation(mf[t])
        R = np.kron(np.diag(vRho), np.identity(iN//iP))
        residual = mY[t] - (R @ mmW[t]).dot(mY[t]) - mmX[t].dot(vBeta).flatten()

        # log likelihood
        vloglik[t] = np.log(np.linalg.det(np.identity(iN) - (R @ mmW[t]))) - iN * 0.5 * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(mSigma)) - 0.5 * residual.dot(np.linalg.inv(mSigma)).dot(residual)

        # generate mf_t+1
        mScores[t] = score(mY[t], mmX[t], mf[t], mmW[t], specification, vBeta, mSigma, vRho)
        mInnovations[t] = scaled_score(mScores[t], specification.scaling_matrix(iP))
        mf[t+1] = vOmega + mA.dot(mInnovations[t]) + mB.dot(mf[t])

    # compute likelihood and raise error
    avg_log_lik = 1/(iT * iN) * np.sum(vloglik)

    return mf, mScores, mInnovations, avg_log_lik