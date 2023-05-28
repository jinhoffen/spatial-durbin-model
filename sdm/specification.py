"""This module provides a hyperparameter class."""

import numpy as np

class Hyperparameter:
    """
    Hyperparameters of the Spatial GAS model

    Parameters
    ----------
    conditional_distribution: str
        String representation of the conditional distribution of y_t. Default value is "mvnorm".
    scaling_matrix: str
        String representation of the scaling matrix S for the conditional score. Default value is "identity".
    gamma_factor: float
        Regulation parameter for the transformation function.
    """
    def __init__(
        self,
        conditional_distribution: str,
        scaling_matrix: str,
        gamma_factor: float
    ) -> None:
        # check arguments
        self._conditional_distribution = conditional_distribution
        self._scaling_matrix = scaling_matrix
        self._gamma_factor = gamma_factor

    @property
    def conditional_distribution(self) -> callable:
        """Draw from the specified conditional distribution"""
        if self._conditional_distribution == "mvnorm":
            return np.random.multivariate_normal

    @conditional_distribution.setter
    def conditional_distribution(self, value) -> None:
        if value in ["mvnorm"]:
            self._conditional_distribution = value
        else:
            raise ValueError("Conditional Distribution is not supported.")

    @property
    def scaling_matrix(self) -> callable:
        """Scaling matrix"""
        if self._scaling_matrix == "identity":
            return np.identity

    @scaling_matrix.setter
    def scaling_matrix(self, value) -> None:
        if value in ["identity"]:
            self._scaling_matrix = value
        else:
            raise ValueError("Scaling is not supported.")

    @property
    def gamma_factor(self) -> float:
        """Regulation parameter for the transformation of the time-varying parameter vector"""
        return self._gamma_factor

    @gamma_factor.setter
    def gamma_factor(self, value) -> None:
        if value > 0 and value < 1:
            self._gamma_factor = value
        else:
            raise ValueError("Gamma factor must be between 0 and 1.")

    @property
    def transformation(self) -> callable:
        """Transformation function of filter"""
        def tanh(f):
            return self._gamma_factor * np.tanh(f)

        return tanh

    @property
    def transformation_inverse(self) -> callable:
        """Transformation function of filter"""
        def tanh_inverse(f):
            return np.log((f + self._gamma_factor)/(self._gamma_factor - f)) * (1/2)
        return tanh_inverse

    @property
    def transformation_foc(self):
        """First derivative of transformation function"""
        def tanh_foc(f):
            return self._gamma_factor * (1 - (np.tanh(f) ** 2))

        return tanh_foc