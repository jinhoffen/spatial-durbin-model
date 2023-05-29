import numpy as np
import sdm
import pytest

def test_shape_consistency():
    """all adjacency matrices must have the same shape"""
    matrices = [
        np.array([
            [0,0,1],
            [1,0,0],
            [1,0,0]
        ]),
        np.array([
            [0,0],
            [1,0]
        ])
    ]

    with pytest.raises(ValueError):
        adjacency_matrix = sdm.AdjacencyMatrix()
        adjacency_matrix.add(0, 10, matrices[0])
        adjacency_matrix.add(10, 20, matrices[1])

def test_isolated_unit():
    """all cross section units must have at least one connection"""
    matrix = np.array([
        [0,0,1],
        [1,0,0],
        [0,0,0]
    ])

    with pytest.raises(ValueError):
        adjacency_matrix = sdm.AdjacencyMatrix()
        adjacency_matrix.add(0, 10, matrix)


def test_self_link():
    """no diagonal element can be different from zero"""
    matrix = np.array([
        [0,0,1],
        [1,1,0],
        [0,1,0]
    ])

    with pytest.raises(ValueError):
        adjacency_matrix = sdm.AdjacencyMatrix()
        adjacency_matrix.add(0, 10, matrix)

def test_row_normalization():
    """test row normalization"""
    adjacency_matrix = np.array([
        [0,1,2],
        [3,0,0],
        [1,1,0]
    ])

    row_normalized = sdm.AdjacencyMatrix.row_normalize(adjacency_matrix)
    expected = np.array([
        [0,1/3,2/3],
        [1,0,0],
        [1/2,1/2,0]
    ])

    assert np.array_equal(row_normalized, expected)

def test_build():
    """test expansion"""
    matrices = [
        np.array([
            [0,1,2],
            [3,0,0],
            [1,1,0]
        ]),
        np.array([
            [0,2,1],
            [1,0,1],
            [1,0,0]
        ])
    ]

    adjacency_matrix = sdm.AdjacencyMatrix()
    adjacency_matrix.add(0, 2, matrices[0])
    adjacency_matrix.add(2, 3, matrices[1])
    weighting_matrix = adjacency_matrix.build()

    # matrix of dimensions 3 x 3 x 3
    expected = np.array([
        [
            [0,1/3,2/3],
            [1,0,0],
            [1/2,1/2,0]
        ],
        [
            [0,1/3,2/3],
            [1,0,0],
            [1/2,1/2,0]
        ],
        [
            [0,2/3,1/3],
            [1/2,0,1/2],
            [1,0,0]
        ]
    ])

    assert np.array_equal(weighting_matrix, expected)