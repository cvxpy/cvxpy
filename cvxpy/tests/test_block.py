import numpy as np
import pytest

import cvxpy as cp


def test_block_scalar_promotion():
    U = cp.Variable((2,1))
    M = cp.block([[4, U.T],
                  [U, np.eye(2)]])
    assert M.shape == (3,3)


def test_block_numpy_behavior():
    v = np.array([1,2,3])
    M = cp.block([[v]])
    assert M.shape == (1,3)


def test_block_pure_scalars():
    M = cp.block([[1,2],[3,4]])
    assert M.shape == (2,2)

def test_block_3d_concat():
    X = cp.Variable((2,2,2))
    Y = cp.block([[X, X]])
    assert Y.shape == (2,2,4)


def test_block_nested_3d():
    X = cp.Variable((2,2,2))
    Y = cp.block([[[X], [X]]])
    assert Y.shape == (2,4,2)

def test_block_4d():
    X = cp.Variable((2,2,2,2))
    Y = cp.block([[X, X], [X, X]])
    assert Y.shape == (2,2,4,4)

def test_block_numpy_equivalence():
    A = np.ones((2,3,4))
    B = np.ones((2,3,4))
    np_result = np.block([[A, B]])
    cp_result = cp.block([[A, B]])
    assert cp_result.shape == np_result.shape

def test_block_3d_row():
    X = cp.Variable((2,2,2))
    Y = cp.block([[X, X]])
    assert Y.shape == (2,2,4)


def test_block_3d_column():
    X = cp.Variable((2,2,2))
    Y = cp.block([[X], [X]])
    assert Y.shape == (2,4,2)


def test_block_mixed_ndim():
    X = cp.Variable((2,2,2))
    with pytest.raises(ValueError):
        cp.block([[X, 1]])

def test_block_deep_nesting():
    X = cp.Variable((2,2,2))
    Y = cp.block([[[[X]]]])
    assert Y.shape == (1, 2, 2, 2)

def test_block_numpy_multiple_patterns():
    A = np.ones((2,3,4))
    B = np.ones((2,3,4))
    C = np.ones((2,3,4))

    np_res = np.block([[A, B], [C, A]])
    cp_res = cp.block([[A, B], [C, A]])

    assert cp_res.shape == np_res.shape

def test_block_invalid_structure():
    A = np.ones((2,3))
    B = np.ones((4,3))
    M = cp.block([[A], [B]])
    assert M.shape == (6,3)



