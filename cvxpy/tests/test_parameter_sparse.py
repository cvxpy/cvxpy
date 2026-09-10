"""
Tests for Issue #2890: cp.Parameter should reject sparse matrix values
at assignment time with a clear ValueError, not silently accept them
and fail later during canonicalization.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp


class TestParameterSparseValueRejection:

    @staticmethod
    def _make_sparse_csc(n: int = 4) -> sp.csc_array:
        return sp.eye(n, format="csc")

    @staticmethod
    def _make_sparse_from_grad() -> sp.spmatrix:
        x = cp.Variable((2, 2), value=[[1, 0], [0, 1]])
        y = x + 1
        return y.grad[x]

    def test_rejects_sparse_csc_at_construction(self):
        with pytest.raises(ValueError, match="sparse"):
            cp.Parameter((4, 4), value=self._make_sparse_csc(4))

    def test_rejects_sparse_csr_at_construction(self):
        with pytest.raises(ValueError, match="sparse"):
            cp.Parameter((3, 3), value=sp.eye(3, format="csr"))

    def test_rejects_sparse_coo_at_construction(self):
        with pytest.raises(ValueError, match="sparse"):
            cp.Parameter((2, 2), value=sp.eye(2, format="coo"))

    def test_rejects_grad_sparse_at_construction(self):
        grad = self._make_sparse_from_grad()
        with pytest.raises(ValueError, match="sparse"):
            cp.Parameter((4, 4), value=grad, name="p")

    def test_rejects_sparse_via_setter(self):
        p = cp.Parameter((4, 4), name="p")
        with pytest.raises(ValueError, match="sparse"):
            p.value = self._make_sparse_csc(4)

    def test_rejects_grad_sparse_via_setter(self):
        p = cp.Parameter((4, 4), name="p")
        with pytest.raises(ValueError, match="sparse"):
            p.value = self._make_sparse_from_grad()

    def test_accepts_dense_ndarray(self):
        p = cp.Parameter((3, 3))
        p.value = np.eye(3)
        assert p.value is not None

    def test_accepts_none(self):
        p = cp.Parameter((3, 3))
        p.value = None
        assert p.value is None

    def test_accepts_scalar(self):
        p = cp.Parameter()
        p.value = 3.14
        assert abs(p.value - 3.14) < 1e-10

    def test_error_mentions_constant_alternative(self):
        p = cp.Parameter((4, 4))
        with pytest.raises(ValueError, match="cp.Constant"):
            p.value = self._make_sparse_csc(4)
