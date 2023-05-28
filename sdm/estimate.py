"""This module provides estimation capabilities."""

import numpy as np
import time
import scipy

from sdm.specification import Hyperparameter

from sdm.starting_values import initial_parameters
from sdm.inference import information_criteria, robust_se, hypothesis_testing
from sdm.utils import flatten_parameters, rebuild_parameters

from sdm import filter, optimizer

class SpatialGasEstimate:
    """
    Estimates the Spatial GAS Model

    Parameters
    ----------
    specification: Hyperparameter
        Model hyperparameters.
    mY: np.ndarray
        Observed data. Dimensions: iT x iN.
    mmX: np.ndarray
        Covariates data. Dimensions: iP x iN x iK.
    mmW: np.ndarray
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.
    iI: int
        Number of groups in first dimension of lexicographically
        sorted, two-dimensional balanced panel with index (i,j).
    """
    def __init__(
        self,
        specification: Hyperparameter,
        mY: np.ndarray,
        mmX: np.ndarray,
        mmW: np.ndarray,
        iI: int
    ):
        self.specification = specification
        self.mY = mY
        self.mmX = mmX
        self.mmW = mmW
        self.iI = iI
        self.model = None

        # infer model size
        self.iT, self.iN, _ = self.mmW.shape
        _, _, self.iK = self.mmX.shape

        # initialize model to be estimated
        self.iP = None
        self.params = None

    def dynamic_heterogenous(self) -> None:
        """estimate dynamic heterogenous gas model"""
        # initial parameters
        params_init = initial_parameters(self.mY, self.mmX, self.mmW, self.iP, self.specification)

        # estimate model
        self.res = scipy.optimize.minimize(
            fun=optimizer.dynamic_heterogenous,
            x0=params_init,
            method="BFGS",
            args=(self.specification, self.mY, self.mmX, self.mmW, self.iP),
            options={
                "disp": False,
                "maxiter": 100000
            }
        )

        # store initial and estimated parameters
        params_init_list = rebuild_parameters(params_init, self.iP, self.iP, self.iP, self.iK, 1)
        for param, label in zip(params_init_list, ["vOmega", "vA", "vB", "vBeta", "dSigma2"]):
            self.params_init[label] = param

        params_list = rebuild_parameters(self.res.x, self.iP, self.iP, self.iP, self.iK, 1)
        for param, label in zip(params_list, ["vOmega", "vA", "vB", "vBeta", "dSigma2"]):
            self.params[label] = param

        # filter
        self.mf, self.mScores, self.mInnovations, self.L = filter.dynamic_heterogenous(self.mY, self.mmX, self.mmW, self.params, self.specification)

        # heteroskedasticity robust standard errors
        self.se = robust_se(
            optimizer.dynamic_heterogenous,
            self.res.hess_inv,
            self.res.x,
            self.specification,
            self.mY,
            self.mmX,
            self.mmW,
            self.iP
        )

    def static_heterogenous(self) -> None:
        """estimate static heterogenous gas model"""
        # initial parameters
        vOmega_init = np.repeat(0.2, self.iP)
        vBeta_init = np.repeat(0.2, self.iK)
        dSigma2_init = np.array([1.0])

        params_init = flatten_parameters(vOmega_init, vBeta_init, dSigma2_init)

        # estimate model
        self.res = scipy.optimize.minimize(
            fun=optimizer.static_heterogenous,
            x0=params_init,
            method="BFGS",
            args=(self.specification, self.mY, self.mmX, self.mmW, self.iP),
            options={
                "disp": False,
                "maxiter": 100000
            }
        )

        # store initial and estimated parameters
        params_init_list = rebuild_parameters(params_init, self.iP, self.iK, 1)
        for param, label in zip(params_init_list, ["vOmega", "vBeta", "dSigma2"]):
            self.params_init[label] = param

        params_list = rebuild_parameters(self.res.x, self.iP, self.iK, 1)
        for param, label in zip(params_list, ["vOmega", "vBeta", "dSigma2"]):
            self.params[label] = param

        # filter
        self.mf, self.L = filter.static_heterogenous(self.mY, self.mmX, self.mmW, self.params, self.specification)
        self.mScores = None
        self.mInnovations = None

        # heteroskedasticity robust standard errors
        self.se = robust_se(
            optimizer.static_heterogenous,
            self.res.hess_inv,
            self.res.x,
            self.specification,
            self.mY,
            self.mmX,
            self.mmW,
            self.iP
        )

    def estimate(self, model: str) -> dict:
        """
        Estimate spatial gas model

        Parameters
        ----------
        model: str
            Type of model to estimate

        Returns
        -------
        res: dict
        """
        self.model = model

        start_time = time.time()

        # estimate user specified type of model
        if model == "dynamic_heterogenous":
            self.iP = self.iI
            self.params = {
                "vOmega": None,
                "vA": None,
                "vB": None,
                "vBeta": None,
                "dSigma2": None
            }
            self.params_init = {
                "vOmega": None,
                "vA": None,
                "vB": None,
                "vBeta": None,
                "dSigma2": None
            }
            self.dynamic_heterogenous()
        elif model == "static_heterogenous":
            self.iP = self.iI
            self.params = {
                "vOmega": None,
                "vBeta": None,
                "dSigma2": None
            }
            self.params_init = {
                "vOmega": None,
                "vBeta": None,
                "dSigma2": None
            }
            self.static_heterogenous()
        else:
            raise ValueError("Type of model is not defined.")

        self.elapsed_time = np.round((time.time() - start_time)/60, 3)

        # hypothesis testing
        self.tval, self.pval = hypothesis_testing(self.res.x, self.se)

        # information criteria
        log_lik = -self.L * self.iT * self.iN
        self.IC = information_criteria(len(self.res.x), self.iT, log_lik)

        # construct result
        return {
            "specification": self.specification,
            "model": self.model,
            "time_units": self.iT,
            "cross_section_units": self.iN,
            "ncovariates": self.iK,
            "gamma_factor": self.specification.gamma_factor,
            "elapsed_time": self.elapsed_time,
            "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "convergence": self.res.success,
            "params": self.params,
            "params_init": self.params_init,
            "hess": None,
            "se": None, #se,
            "tval": None, #tval,
            "pval": None, #pval,
            "aic": self.IC["AIC"],
            "bic": self.IC["BIC"],
            "mY": self.mY,
            "mmX": self.mmX,
            "mmW": self.mmW,
            "mf": self.mf,
            "mScores": self.mScores,
            "mInnovations": self.mInnovations,
            "mRho": self.specification.transformation(self.mf),
            "log_lik": log_lik
        }