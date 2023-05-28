"""This module determines starting values for diagonal elements of A and B matrix."""

import numpy as np

import scipy

from sdm import utils
from sdm.optimizer import static_heterogenous, dynamic_heterogenous

def initial_parameters(
    mY,
    mmX,
    mmW,
    iP,
    specification
):
    """Define initial parameters for the dynamic heterogenous model"""
    # parse parameters
    iK = mmX.shape[2]

    # estimate static GAS model
    vMu_init = np.repeat(0.2, iP)
    vBeta_init = np.repeat(0.2, iK)
    dSigma2_init = np.array([1.0])

    params_init = utils.flatten_parameters(vMu_init, vBeta_init, dSigma2_init)
    res = scipy.optimize.minimize(
        fun=static_heterogenous,
        x0=params_init,
        method="BFGS",
        args=(specification, mY, mmX, mmW, iP),
        options={
            "disp": False,
            "maxiter": 100000
        }
    )

    # set initial values for vOmega, vBeta, dSigma2 based on fitted static GAS model parameters
    # set mB to some diagonal matrix
    # perform grid search for mA
    vA_init = initial_mA(res.x, mY, mmX, mmW, iP, specification)

    # set initial values of vOmega, vBeta, dSigma2 based on fitted static GAS model parameters
    # set vA to result from previous grid search
    # perform grid search for mB
    vB_init = initial_mB(res.x, vA_init, mY, mmX, mmW, iP, specification)

    # use estimated parameters from above procedure to construct final initial parameters
    vMu_init, vBeta_init, dSigma2_init = utils.rebuild_parameters(res.x, iP, iK, 1)
    vOmega_init = vMu_init.dot(np.eye(iP)-np.diag(vB_init))
    params_init = np.concatenate((vOmega_init, vA_init, vB_init, vBeta_init, dSigma2_init))

    return params_init

def initial_mA(
    params_init,
    mY,
    mmX,
    mmW,
    iP,
    specification
):
    """Perform grid search for on diagonal elements of A matrix"""
    # parse parameters
    iK = mmX.shape[2]

    # rebuild parameters
    vMu_init, vBeta_init, dSigma2_init = utils.rebuild_parameters(params_init, iP, iK, 1)

    # fix vB
    vB_init = np.ones(iP) * 0.9
    vOmega_init = vMu_init.dot(np.eye(iP) - np.diag(vB_init))

    # initialize variables
    dloglik_best = np.inf
    seq_vA = np.linspace(1e-04, 0.9, 100)

    # initialize vA, update array when better likelihood value is found
    vA = np.ones(iP) * 0.01

    # treat diagonal elements independently because they are uncorrelated
    for i in range(iP):
        for j in range(len(seq_vA)):
            # safe current
            fallback_value = vA[i]
            # update vA with new try
            vA[i] = seq_vA[j]
            # construct parameter array
            params_seq = np.concatenate((vOmega_init, vA, vB_init, vBeta_init, dSigma2_init))
            # compute log likelihood
            dloglik = dynamic_heterogenous(params_seq, specification, mY, mmX, mmW, iP)
            # update if log likelihood is better
            # account for the fact that optimiser returns negative value due to the usage of minimization algorithm
            if (dloglik < dloglik_best):
                dloglik_best = dloglik
                vA[i] = seq_vA[j]
            else:
                vA[i] = fallback_value

    return vA

def initial_mB(
    params_init,
    vA_init,
    mY,
    mmX,
    mmW,
    iP,
    specification
):
    """Perform grid search for on diagonal elements of B matrix"""
    # parse parameters
    iK = mmX.shape[2]

    vMu_init, vBeta_init, dSigma2_init = utils.rebuild_parameters(params_init, iP, iK, 1)

    # initialize variables
    dloglik_best = np.inf
    seq_vB = np.linspace(1e-04, 0.9, 100)

    # initialize vB with zeros, update array when better likelihood value is found
    vB = np.ones(iP)*0.9

    for i in range(iP):
        for j in range(len(seq_vB)):
            # safe current
            fallback_value = vB[i]
            # update vA with new try
            vB[i] = seq_vB[j]
            # update vOmega_init based on new try
            vOmega_init = vMu_init.dot(np.eye(iP)-np.diag(vB))
            # construct parameter array
            params_seq = np.concatenate((vOmega_init, vA_init, vB, vBeta_init, dSigma2_init))
            # compute log likelihood
            dloglik = dynamic_heterogenous(params_seq, specification, mY, mmX, mmW, iP)
            # update if log likelihood is better
            # account for the fact that optimiser returns negative value due to the usage of minimization algorithm
            if (dloglik < dloglik_best):
                dloglik_best = dloglik
                vB[i] = seq_vB[j]
            else:
                vB[i] = fallback_value

    return vB