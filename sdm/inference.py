"""This module computes the robust standard errors and information criteria."""

import numpy as np
import scipy

def robust_se(
    likelihood: callable,
    mInvHessian: np.ndarray,
    params: np.ndarray,
    gamma_factor: float,
    mY: np.ndarray,
    mmX: np.ndarray,
    mmW: np.ndarray,
    iP: int
) -> np.ndarray:
    r"""
    Huber-White Sandwich Estimator / Heteroskedasticity robust standard errors

    Heteroskedasticity Consistent Covariance Estimator (HC0) by White (1980):

    .. math::

        \sqrt\left{(X^T X)^{-1} X.T \diag[\epsilon^2_i] X (X^T X)^{-1} \right}

    Parameters
    ----------
    likelihood: callable
        Likelihood function
    mInvHessian: np.ndarray
        Approximation of the Hessian inverse from the Optimizer minimization.
    params: : np.ndarray
        Parameter vector.
    gamma_factor: float
        Regulation parameter for the transformation of the time-varying parameter vector f.
    mY: np.ndarray
        Observation data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iT x iN x iK.
    mmW: np.ndarray
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.

    Notes
    -----
    See `On The So-Called Huber Sandwich Estimator and Robust Standard Errors <https://www.stat.berkeley.edu/~census/mlesan.pdf/>`_
    and `Statsmodels <https://www.statsmodels.org/dev/_modules/statsmodels/stats/sandwich_covariance.html#cov_hc0/>`.
    """
    # numerically approximate the gradient
    eps = np.sqrt(np.finfo(float).eps)
    epsilon = np.repeat(eps, len(params))
    vJacobian = scipy.optimize.approx_fprime(params, likelihood, epsilon, gamma_factor, mY, mmX, mmW, iP)

    # meat matrix or outer product of gradients
    outer_product_gradient = (vJacobian[np.newaxis].T).dot(vJacobian[np.newaxis])

    # compute sandwich variance estimator
    mSandwich = (mInvHessian).dot(outer_product_gradient).dot(mInvHessian)

    return np.sqrt(np.diag(mSandwich))

def hypothesis_testing(
    params: np.ndarray,
    vSE: np.ndarray
) -> list:
    """Compute t statistic and p value"""
    vTval = params/vSE
    vPval = 2*(1 - scipy.stats.norm.cdf(np.abs(vTval)))

    return vPval, vTval

def information_criteria(
    num_params: int,
    sample_size: int,
    loglik: float
):
    """
    Construct information criteria

    Parameters
    ----------
    num_params : int
        number of estimated parameters
    sample_size : int
        sample size (number of time series observations)
    loglik : float
        log likelihood value

    Returns
    -------
    dict
        Information Criteria
    """
    AIC = 2 * (num_params - loglik)
    BIC = -2 * loglik + num_params * np.log(sample_size)

    return { 'AIC': AIC, 'BIC': BIC }