"""This module provides simulation functionalities for covariates."""

import numpy as np

def simulate(
    iT: int,
    iN: int,
    iK: int
) -> np.ndarray:
    """
    Simulate covariates from multivariate normal distribution

    The function returns a iT x iN x iK matrix of simulated
    covariates from the multivariate normal distribution.

    Parameters
    ----------
    iT: int
        Number of time periods.
    iN: int
        Number of units j per group i in two-dimensional panel index.
    iK: int
        Number of variables to simulate.
    """
    mmX = np.random.multivariate_normal(
        mean=np.zeros(iN),
        cov=np.identity(iN),
        size=(iT, iK)
    )

    return np.transpose(mmX, (0, 2, 1))