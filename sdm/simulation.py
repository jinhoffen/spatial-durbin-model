"""This module contains simulation results."""

import numpy as np

from sdm.utils import score, scaled_score
from sdm.specification import Hyperparameter
from sdm.results import SimulationResult

class SimulateModel:
    """
    Run Simulation for the Spatial GAS model

    Parameters
    ----------
    specification: Hyperparameter
        Model hyperparameters.
    mmX: np.ndarray
        Covariates data. Dimensions: iT x iN x iK.
    mmW: np.ndarray
        Row-normalized weighting matrix. Dimensions: iT x iN x iN.
    params: dict
        Parameter dictionary.
    """
    def __init__(
        self,
        specification: Hyperparameter,
        mmX: np.ndarray,
        mmW: list,
        params_filter: dict,
        params_observation: dict,
        process: str
    ) -> None:
        # arguments
        self.specification = specification
        self.mmX = mmX
        self.mmW = mmW
        self.params_observation = params_observation
        self.params_filter = params_filter

        self.process = process

        # parameter
        self.vBeta = params_observation["vBeta"]

        # implications
        self.iT, self.iN, _ = self.mmW.shape
        self.iK = self.mmX.shape[2]

        if process in ["autoregressive", "static"]:
            self.iP = len(params_filter["vOmega"])
        elif process == "step":
            self.iP = len(params_filter["vOmega1"])
        elif process == "ramp":
            self.iP = len(params_filter["vBreakpoint"])
        else:
            raise ValueError("Process is not supported.")

        self.mSigma = np.identity(self.iN) * params_observation["dSigma2"]

    def autoregressive_process(
        self,
        mEps: np.ndarray
    ):
        """Simulate filter with autoregressive process"""
        # parse parameters
        vOmega = self.params_filter["vOmega"]
        mA = np.diag(self.params_filter["vA"])
        mB = np.diag(self.params_filter["vB"])

        # initialize observation, score, innovation, and time varying parameter vector
        mY = np.zeros((self.iT, self.iN))
        mF = np.zeros((self.iT+1, self.iP))
        mScores = np.zeros((self.iT, self.iP))
        mInnovations = np.zeros((self.iT, self.iP))

        # initialize dynamics
        mF[0] = np.linalg.inv(np.identity(self.iP) - mB).dot(vOmega)

        # simulate from data generating process
        for t in range(self.iT):
            # generate observation y_t
            vRho = self.specification.transformation(mF[t])
            R = np.kron(np.diag(vRho), np.identity(self.iN//self.iP))
            Zt = np.linalg.inv(np.identity(self.iN) - (R @ self.mmW[t]))
            mY[t] = Zt.dot(self.mmX[t].dot(self.vBeta) + mEps[t])

            # update time varying parameter
            mScores[t] = score(mY[t], self.mmX[t], mF[t], self.mmW[t], self.specification, self.vBeta, self.mSigma, vRho)
            mInnovations[t] = scaled_score(mScores[t], self.specification.scaling_matrix(self.iP))
            mF[t+1] = vOmega + mA.dot(mInnovations[t]) + mB.dot(mF[t])

        return mF, mScores, mInnovations, mY

    def static_process(self, mEps):
        """Simulate filter with static process"""
        # parse parameters
        static_rho = self.params_filter["vOmega"]

        # initialize the time varying parameter
        mF = np.zeros((self.iT+1, self.iP))
        mF[0] = self.specification.transformation_inverse(static_rho)

        # initialize observation, score and innovation vector
        mY = np.zeros((self.iT, self.iN))

        # simulate time series from DGP
        for t in range(self.iT):
            # generate observation y_t
            vRho = self.specification.transformation(mF[t])
            R = np.kron(np.diag(vRho), np.identity(self.iN//self.iP))
            Zt = np.linalg.inv(np.identity(self.iN) - (R @ self.mmW[t]))
            mY[t] = Zt.dot(self.mmX[t].dot(self.vBeta) + mEps[t])

            # generate f_t+1
            mF[t+1] = self.specification.transformation_inverse(static_rho)

        return mF, mY

    def step_process(self, mEps):
        """Simulate filter with step process"""
        # parse parameters
        vOmega1 = self.params_filter["vOmega1"]
        vOmega2 = self.params_filter["vOmega2"]

        # define step function for spatial dependence parameter
        step_rho = lambda t, vOmega1, vOmega2: np.array(vOmega1) if t>self.iT/2 else np.array(vOmega2)

        # initialize the time varying parameter
        mF = np.zeros((self.iT+1, self.iP))
        vRho_init = step_rho(0, vOmega1, vOmega2)
        mF[0] = self.specification.transformation_inverse(vRho_init)

        # initialize observation, score and innovation vector
        mY = np.zeros((self.iT, self.iN))

        # simulate time series from DGP
        for t in range(self.iT):
            # generate observation y_t
            vRho = self.specification.transformation(mF[t])
            R = np.kron(np.diag(vRho), np.identity(self.iN//self.iP))
            Zt = np.linalg.inv(np.identity(self.iN) - (R @ self.mmW[t]))
            mY[t] = Zt.dot(self.mmX[t].dot(self.vBeta) + mEps[t])

            # generate f_t+1
            vRho_step = step_rho(t, vOmega1, vOmega2)
            mF[t+1] = self.specification.transformation_inverse(vRho_step)

        return mF, mY

    def ramp_process(self, mEps):
        """Simulate filter with ramp process"""
        # parse parameters
        vBreakpoint = self.params_filter["vBreakpoint"]
        vBounds = self.params_filter["vBounds"]

        # define ramp process for spatial dependence parameter
        ramp_rho = lambda t, iT, breakpoints, bound: (bound / (iT // breakpoints) * t) % bound

        # initialize the time varying parameter
        mF = np.zeros((self.iT+1, self.iP))
        vRho_init = ramp_rho(0, self.iT, vBreakpoint, vBounds)
        mF[0] = self.specification.transformation_inverse(vRho_init)

        # initialize observation, score and innovation vector
        mY = np.zeros((self.iT, self.iN))

        # simulate time series from DGP
        for t in range(self.iT):
            # generate observation y_t
            vRho = self.specification.transformation(mF[t])
            R = np.kron(np.diag(vRho), np.identity(self.iN//self.iP))
            Zt = np.linalg.inv(np.identity(self.iN) - (R @ self.mmW[t]))
            mY[t] = Zt.dot(self.mmX[t].dot(self.vBeta) + mEps[t])

            # update time varying parameter
            vRho_ramp = ramp_rho(t, self.iT, vBreakpoint, vBounds)
            mF[t+1] = self.specification.transformation_inverse(vRho_ramp)

        return mF, mY

    def simulate(self) -> SimulationResult:
        """Simulate the spatial model"""
        # initialize results
        res = {
            "specification": self.specification,
            "params_observation": self.params_filter,
            "params_filter": self.params_filter,
            "time_units": self.iT,
            "cross_section_units": self.iN,
            "ncovariates": self.iK,
            "nparameters": self.iP,
            "mmX": self.mmX,
            "mmW": self.mmW,
        }

        # construct errors of dimension iT x iN
        mEps = np.random.multivariate_normal(mean=np.zeros(self.iN), cov=self.mSigma, size=self.iT)

        # define process
        if self.process == "autoregressive":
            mF, mScores, mInnovations, mY = self.autoregressive_process(mEps)
            res["mScores"] = mScores
            res["mInnovations"] = mInnovations
        elif self.process == "static":
            mF, mY = self.static_process(mEps)
            res["mScores"] = None
            res["mInnovations"] = None
        elif self.process == "step":
            mF, mY = self.step_process(mEps)
            res["mScores"] = None
            res["mInnovations"] = None
        elif self.process == "ramp":
            mF, mY = self.ramp_process(mEps)
            res["mScores"] = None
            res["mInnovations"] = None
        else:
            raise ValueError("This process is not defined.")

        # add simulated observations and filter
        res["mY"] = mY
        res["mf"] = mF
        res["mRho"] = self.specification.transformation(mF)

        return SimulationResult(res)