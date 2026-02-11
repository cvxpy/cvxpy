import numpy as np

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
