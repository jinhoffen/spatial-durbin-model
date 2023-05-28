"""This module provides results containers."""

import pickle
import numpy as np
import pandas as pd

from sdm.specification import Hyperparameter
from sdm.utils import flatten_parameters


class SimulationResult:
    """
    Results container for simulated model
    """
    def __init__(self, res: dict) -> None:
        self._specification = res["specification"]
        self._params_observation = res["params_observation"]
        self._params_filter = res["params_filter"]
        self._time_units = res["time_units"]
        self._cross_section_units = res["cross_section_units"]
        self._ncovariates = res["ncovariates"]
        self._nparameters = res["nparameters"]
        self._mY = res["mY"]
        self._mmX = res["mmX"]
        self._mmW = res["mmW"]
        self._mf = res["mf"]
        self._mScores = res["mScores"]
        self._mInnovations = res["mInnovations"]
        self._mRho = res["mRho"]

    @property
    def specification(self) -> Hyperparameter:
        """Model hyperparameter"""
        return self._specification

    @property
    def params_observation(self) -> dict:
        """Parameters of the observation equation"""
        return self._params_observation

    @property
    def params_filter(self) -> dict:
        """Parameters of the updating equation"""
        return self._params_filter

    @property
    def time_units(self) -> int:
        """Number of time units"""
        return self._time_units

    @property
    def cross_section_units(self) -> int:
        """Number of units per group in balanced panel"""
        return self._cross_section_units

    @property
    def ncovariates(self) -> int:
        """Number of covariates"""
        return self._ncovariates

    @property
    def nparameters(self) -> int:
        """Number of simulated parameters"""
        return self._nparameters

    @property
    def mY(self) -> np.ndarray:
        """Simulated observations"""
        return self._mY

    @property
    def mmX(self) -> np.ndarray:
        """Covariates"""
        return self._mmX

    @property
    def mf(self) -> np.ndarray:
        """Time-varying parameter vector"""
        return self._mf

    @property
    def mScores(self) -> np.ndarray:
        """Score of the conditional density"""
        return self._mScores

    @property
    def Innovations(self) -> np.ndarray:
        """Innovation or scaled score of the conditional density"""
        return self._mInnovations

    @property
    def mRho(self) -> np.ndarray:
        """Spatial dependence parameter"""
        return self._mRho


class EstimationResult:
    """
    Results container for estimated model
    """
    def __init__(self, res: dict) -> None:
        self._specification = res["specification"]
        self._model = res["model"]
        self._time_units = res["time_units"]
        self._cross_section_units = res["cross_section_units"]
        self._ncovariates = res["ncovariates"]
        self._gamma_factor = res["gamma_factor"]
        self._elapsed_time = res["elapsed_time"]
        self._execution_date = res["execution_date"]
        self._convergence = res["convergence"]
        self._params = res["params"]
        self._params_init = res["params_init"]
        self._hess = res["hess"]
        self._se = res["se"]
        self._tval = res["tval"]
        self._pval = res["pval"]
        self._aic = res["aic"]
        self._bic = res["bic"]
        self._mY = res["mY"]
        self._mmX = res["mmX"]
        self._mmW = res["mmW"]
        self._mf = res["mf"]
        self._mScores = res["mScores"]
        self._mInnovations = res["mInnovations"]
        self._mRho = res["mRho"]
        self._log_lik = res["log_lik"]

    @property
    def specification(self) -> Hyperparameter:
        """Model hyperparameter"""
        return self._specification

    @property
    def model(self) -> str:
        """Estimated model"""
        return self._model

    @property
    def time_units(self) -> int:
        """Number of time units"""
        return self._time_units

    @property
    def cross_section_units(self) -> int:
        """Number of units per group in balanced panel"""
        return self._cross_section_units

    @property
    def ncovariates(self) -> int:
        """Number of covariates"""
        return self._ncovariates

    @property
    def gamma_factor(self) -> int:
        """Regulation parameter"""
        return self._gamma_factor

    @property
    def elapsed_time(self) -> float:
        """Total and average elapsed time"""
        return self._elapsed_time

    @property
    def execution_date(self) -> str:
        """Datetime of estimation execution"""
        return self._execution_date

    @property
    def convergence(self) -> bool:
        """Boolean indicating whether convergence was successful"""
        return self._convergence

    @property
    def params(self) -> dict:
        """Estimated parameters"""
        return self._params

    @property
    def params_init(self) -> dict:
        """Initial parameters"""
        return self._params_init

    @property
    def hess(self) -> np.ndarray:
        """Hessian"""
        return self._hess

    @property
    def se(self) -> np.ndarray:
        """White robust standard errors"""
        return self._se

    @property
    def tval(self) -> np.ndarray:
        """T values"""
        return self._tval

    @property
    def pval(self) -> np.ndarray:
        """P values"""
        return self._pval

    @property
    def aic(self) -> float:
        """Akike information criterion"""
        return self._aic

    @property
    def bic(self) -> float:
        """Bayesian information criterion"""
        return self._bic

    @property
    def mY(self) -> np.ndarray:
        """Observations"""
        return self._mY

    @property
    def mmX(self) -> np.ndarray:
        """Covariates"""
        return self._mmX

    @property
    def mmW(self) -> np.ndarray:
        """Row-normalized weighting matrix"""
        return self._mmW

    @property
    def mf(self) -> np.ndarray:
        """Time-varying parameter vector"""
        return self._mf

    @property
    def mScores(self) -> np.ndarray:
        """Score of the conditional density"""
        return self._mScores

    @property
    def mInnovation(self) -> np.ndarray:
        """Innovation or scaled score of the conditional density"""
        return self._mInnovation

    @property
    def mRho(self) -> np.ndarray:
        """Spatial dependence parameter"""
        return self._mRho

    @property
    def log_lik(self) -> np.ndarray:
        """Log likelihood function value of estimated model"""
        return self._log_lik

class MonteCarlo:
    """
    Results container for a monte carlo experiment

    Parameters
    ----------
    params_observation: dict
        Parameters of observation equation.
    params_filter: dict
        Parameters of filter equation.
    estimation_model: str
        String representation of the estimated model.
    """
    def __init__(
        self,
        params_observation: dict,
        params_filter: dict,
        estimation_model: str
    ):
        # arguments
        self._params_dgp_observation = params_observation
        self._params_dgp_filter = params_filter
        self._estimation_model = estimation_model

        # initialize
        self._params_est = []
        self._mRho_est = []
        self._mRho_sim = []
        self._convergence = []
        self._time_elapsed = []
        self._nparams = None
        self._labels = None

    @property
    def size(self) -> int:
        """Monte Carlo sample size"""
        return len(self._convergence)

    @property
    def convergence(self) -> np.ndarray:
        """Boolean indicating whether convergence was successful"""
        return self._convergence

    @property
    def params_est(self) -> pd.DataFrame:
        """Estimated parameters in pandas frame"""
        return self.params_est_frame()

    @property
    def params_dgp(self) -> pd.DataFrame:
        """Parameter of data generating process"""
        return {**self._params_dgp_filter, **self._params_dgp_observation}

    @property
    def time_elapsed(self) -> None:
        """Compute total and average elapsed time"""
        total_hours = int(np.sum(self._time_elapsed)//60)
        total_minutes = int(np.sum(self._time_elapsed)%60)
        mean_minutes = int(np.mean(self._time_elapsed))

        msg = (
            f"Total time elapsed: {total_hours}h {total_minutes:02d}min\n"
            f"Average time per estimation: {mean_minutes} min"
        )

        print(msg)

    @property
    def nparams(self) -> int:
        """Number of parameters"""
        return self._nparams

    @property
    def rho_simulated(self):
        """Simulated spatial dependence parameter"""
        return self._mRho_sim

    @property
    def rho_estimated(self):
        """Estimated spatial dependence parameter"""
        return self._mRho_est

    @staticmethod
    def from_file(filename):
        """read experiment from pickle file"""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def to_file(self, filename) -> None:
        """write object on pickle file"""
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def params_est_frame(self) -> pd.DataFrame:
        """Parameter Estimates"""
        result = np.stack(self._params_est)
        index = pd.MultiIndex.from_tuples(self._labels)

        return pd.DataFrame(result, columns=index)

    def add(
        self,
        params_est: dict,
        mRho_sim: np.ndarray,
        mRho_est: np.ndarray,
        convergence: bool,
        time_elapsed: float
    ) -> None:
        """Add monte carlo sample"""
        # parse estimated parameters
        params_list = list(params_est.values())
        params_est_flattened = flatten_parameters(*params_list)

        # store number of parameters
        if self._nparams is None:
            self._nparams = len(params_est_flattened)

        # store parameter labels
        if self._labels is None:
            labels = list(params_est.keys())
            self._labels = [(label, i) for label in labels for i in range(len(params_est[label]))]

        # store spatial dependence parameter
        self._mRho_sim.append(mRho_sim)
        self._mRho_est.append(mRho_est)

        # store converge success result
        self._convergence.append(convergence)

        self._params_est.append(params_est_flattened)

        # store time elapsed
        self._time_elapsed.append(time_elapsed)