"""This module provides the API for the spatialgasmodel package."""

import numpy as np

from sdm.simulation import SimulateModel
from sdm.specification import Hyperparameter
from sdm.estimate import SpatialGasEstimate
from sdm.results import EstimationResult

class SpatialGasModel:
    r"""
    Generalized Autoregressive Score Model

    Parameters
    ----------
    conditional_distribution: str
        String representation of the conditional distribution of y_t. Default value is "mvnorm".
    scaling_matrix: str
        String representation of the scaling matrix S for the conditional score. Default value is "identity".
    gamma_factor: int
        Regulation parameter that ensures the spatial dependence parameter stays in the unity bound.

    Notes
    -----
    The model is given by

    .. math::

        y_t = R_t W_t y_t + X_t \beta + e_t

    where :math:`R_t = \rho_t \otimes \mathbf{I}_N` with :math:`\rho_t` a
    :math:`P` dimensional array and :math:`\mathbf{I}_N` an :math:`N`-dimensional identity matrix.
    :math:`P` and :math:`N` refer to the number of units per dimension in the balanced panel index.
    """

    def __init__(
        self,
        conditional_distribution: str,
        scaling_matrix: str,
        gamma_factor: float
    ) -> None:
        self.specification = Hyperparameter(conditional_distribution, scaling_matrix, gamma_factor)

    def simulate(
        self,
        mmX: np.ndarray,
        mmW: np.ndarray,
        params_observation: dict,
        params_filter: dict,
        process: str
    ) -> SimulateModel:
        """
        Simulate the Spatial Generalized Autoregressive Score Model

        Parameters
        ----------
        mmX: np.ndarray
            Covariates data. Dimensions: iT x iN x iK.
        mmW: np.ndarray
            Row-normalized weighting matrix. Dimensions: iT x iN x iN.
        params_observation: dict
            Parameters of the observation equation.
        params_filter: dict
            Parameters of the updating equation.
        process: str
            String representation of the updating equation.

        Returns
        -------
        SimulationResult
            See :class:`~results.SimulationResult`
        """
        simulation = SimulateModel(self.specification, mmX, mmW, params_observation, params_filter, process)

        return simulation.simulate()

    def estimate(
        self,
        mY: np.ndarray,
        mmX: np.ndarray,
        mmW: np.ndarray,
        cross_section_units: int,
        model: str
    ) -> EstimationResult:
        """
        Estimate the Spatial Generalized Autoregressive Score Model

        Parameters
        ----------
        mY: np.ndarray
            Observation data. Dimensions: iT x iN.
        mmX: np.ndarray
            Covariates data. Dimensions: iT x iN x iK.
        mmW: np.ndarray
            Row-normalized weighting matrix. Dimensions: iT x iN x iN.
        cross_section_units: int
            Number of the cross sectional units j per group i
            in the balanced panel index (i,j).
        model: str
            String representation of the updating equation.

        Returns
        -------
        EstimationResult
            See :class:`~results.EstimationResult`
        """
        estimation = SpatialGasEstimate(self.specification, mY, mmX, mmW, cross_section_units)
        res = estimation.estimate(model)

        return EstimationResult(res)