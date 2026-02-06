import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sparla

import cvxpy as cp
import cvxpy.settings as s
from cvxpy import psd_wrap


def test_is_psd() -> None:

    n = 50

    # trivial cases
    psd = np.eye(n)
    nsd = -np.eye(n)

    assert cp.Constant(psd).is_psd()
    assert not cp.Constant(psd).is_nsd()

    assert cp.Constant(nsd).is_nsd()
    assert not cp.Constant(nsd).is_psd()

    # Test that is_psd works for random PSD matrices.
    # ArpackNoConvergence can occur for numerical reasons on some matrices;
    # when it does, verify the error message is helpful.
    for seed in range(95, 100):
        np.random.seed(seed)

        P = np.random.randn(n, n)
        P = P.T @ P

        try:
            assert cp.Constant(P).is_psd()
        except sparla.ArpackNoConvergence as e:
            assert "CVXPY note" in str(e)

    assert psd_wrap(cp.Constant(P)).is_psd()


def test_is_psd_arpack_no_convergence(monkeypatch) -> None:
    """Test that ArpackNoConvergence is augmented with a helpful message."""
    n = 10

    def mock_eigsh(*args, **kwargs):
        raise sparla.ArpackNoConvergence("mock failure", np.array([]), np.array([[]]))

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)

    # Construct a matrix that bypasses the Gershgorin and diagonal fast paths
    # so it reaches the ARPACK code path.
    A = np.full((n, n), 0.5)
    np.fill_diagonal(A, 1.0)

    with pytest.raises(sparla.ArpackNoConvergence, match="CVXPY note"):
        cp.Constant(A).is_psd()


def test_print():
    A = cp.Constant(np.ones((3, 3)))
    assert str(A) == '[[1.00 1.00 1.00]\n [1.00 1.00 1.00]\n [1.00 1.00 1.00]]'
    B = cp.Constant(np.ones((5, 2)))
    assert str(
        B) == '[[1.00 1.00]\n [1.00 1.00]\n ...\n [1.00 1.00]\n [1.00 1.00]]'
    default = s.PRINT_EDGEITEMS
    s.PRINT_EDGEITEMS = 10
    assert str(
        B) == '[[1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]\n [1.00 1.00]]'
    s.PRINT_EDGEITEMS = default


def test_prod():
    rows = np.concatenate([np.arange(100), np.zeros(100)[1:]])
    cols = np.concatenate([np.zeros(100), np.arange(100)[1:]])
    values = np.ones(199)
    A = sp.coo_matrix((values, (rows, cols)), shape=(100, 100))

    assert np.allclose(cp.prod(A).value, 0.0)
    assert np.allclose(cp.prod(A, axis=0).value, [1] + [0] * 99)
    assert cp.prod(A, axis=0).shape == (100,)
    assert np.allclose(cp.prod(A, axis=1).value, [1] + [0] * 99)
    assert cp.prod(A, axis=1).shape == (100,)
    assert np.allclose(cp.prod(A, axis=0, keepdims=True).value, [[1] + [0] * 99])
    assert cp.prod(A, axis=0, keepdims=True).shape == (1, 100)
    assert np.allclose(cp.prod(A, axis=1, keepdims=True).value, [[1]] + [[0]] * 99)
    assert cp.prod(A, axis=1, keepdims=True).shape == (100, 1)

    B = np.arange(4).reshape(2, 2) + 1
    assert np.allclose(cp.prod(sp.coo_matrix(B)).value, 24)


def test_nested_lists():

    A = [[1, 2], [3, 4], [5, 6]]

    numpy_array = np.array(A)
    constant_from_numpy = cp.Constant(numpy_array)

    with pytest.warns(match="nested list is undefined behavior"):
        constant_from_lists = cp.Constant(A)

    assert np.allclose(constant_from_numpy.value, numpy_array)

    # CVXPY behaviour currenlty is different from NumPy for nested lists,
    # with the order being reversed.
    assert np.allclose(constant_from_lists.value.T, numpy_array)