"""This module converts adjacency to weighting matrices."""

import numpy as np

class AdjacencyMatrix:
    """
    Converts Adjacency Matrix to Weighting Matrices

    Dimensions are iT x iP x iN x iN.
    """
    def __init__(self) -> None:
        self.matrices = []
        self.dimension = None

    @staticmethod
    def row_normalize(matrix):
        """create weighting matrix from adjacency matrix"""
        row_sum = matrix.sum(axis=1)
        return matrix / row_sum[:, np.newaxis]

    @staticmethod
    def check_square_matrix(matrix):
        """check whether adjacency matrix is square"""
        shape = matrix.shape
        if len(shape) != 2:
            return True
        elif shape[0] != shape[1]:
            return True
        else:
            return False

    @staticmethod
    def check_self_connection(matrix):
        """check matrix diagonal for non zero elements"""
        diagonal = matrix.diagonal()
        return any(elem!=0 for elem in diagonal)

    @staticmethod
    def check_isolated_unit(matrix):
        """check whether all cross section units have at least one connection"""
        row_sum = np.sum(matrix, axis=1)
        return any(row==0 for row in row_sum)

    def add(
        self,
        period_start: int,
        period_end: int,
        adjacency_matrix: np.ndarray
    ) -> None:
        """add adjacency matrix"""
        if self.check_isolated_unit(adjacency_matrix):
            raise ValueError("There is at least one cross-section unit without any connection.")
        elif self.check_self_connection(adjacency_matrix):
            raise ValueError("Diagonal elements must all be zero.")
        elif self.check_square_matrix(adjacency_matrix):
            raise ValueError("The matrix must be a square 2d array.")
        elif self.check_adjacency_matrix_shapes(adjacency_matrix):
            raise ValueError("The shape of the adjacency matrix is inconsistent with other matrices.")

        # create new matrix
        weighting_matrix = self.row_normalize(adjacency_matrix)
        new_matrix = (period_start, period_end, adjacency_matrix, weighting_matrix)

        # store new matrix and ensure ordering
        self.matrices.append(new_matrix)
        sorted(self.matrices, key=lambda x: x[0])

    def check_adjacency_matrix_shapes(self, matrix):
        """check whether dimension is consistent with previous adjacency matrices"""
        if self.dimension is None:
            self.dimension = matrix.shape
            return False
        else:
            return matrix.shape != self.dimension

    def build(self):
        """build iT x iP x iN x iN matrix"""
        # expand across time periods
        # create iT x iP x iN x iN matrix
        result = []
        for matrix in self.matrices:
            start, end, _, weighting_matrix = matrix
            length = end - start
            new = [weighting_matrix]*length
            result.extend(new)

        return np.stack(result)