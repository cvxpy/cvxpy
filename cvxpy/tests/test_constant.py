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


def _dense_psd_with_large_off_diagonals(n: int) -> np.ndarray:
    """A positive definite matrix whose Gershgorin check is inconclusive."""
    A = np.full((n, n), 0.5)
    np.fill_diagonal(A, 1.0)
    return A


def test_is_psd_dense_uses_pivoted_cholesky(monkeypatch) -> None:
    """Dense input is certified with pstrf instead of the iterative solver."""
    import cvxpy.utilities.linalg as linalg

    calls: list[tuple[int, ...]] = []
    original = linalg._PSTRF_FUNCTIONS[np.float64]

    def counting_pstrf(a):
        calls.append(a.shape)
        return original(a)

    monkeypatch.setitem(linalg._PSTRF_FUNCTIONS, np.float64, counting_pstrf)

    assert cp.Constant(_dense_psd_with_large_off_diagonals(8)).is_psd()
    assert calls, "the dense path must go through the pivoted Cholesky"


def test_is_psd_dense_indefinite_is_rejected() -> None:
    """pstrf reports a non-positive pivot and the fallback confirms it is indefinite."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert not cp.Constant(A).is_psd()


def test_is_psd_dense_singular_is_accepted() -> None:
    """A singular PSD matrix stops pstrf, and the fallback still certifies it."""
    v = np.array([1.0, 2.0, 3.0])
    E = np.eye(3) - np.outer(v, v) / np.inner(v, v)
    A = E @ np.diag([2.0, 3.0, 4.0]) @ E.T
    assert np.linalg.matrix_rank(A) == 2
    assert cp.Constant(A).is_psd()


def test_is_psd_dense_float32_is_accepted() -> None:
    """The float32 LAPACK wrapper is dispatched to as well."""
    A = _dense_psd_with_large_off_diagonals(8).astype(np.float32)
    assert cp.Constant(A).is_psd()


def test_is_psd_warns_for_large_dense(monkeypatch) -> None:
    """Order >= 5000 warns and points at psd_wrap."""
    import cvxpy.utilities.linalg as linalg

    monkeypatch.setattr(linalg, "_PSTRF_WARN_ORDER", 5)
    with pytest.warns(UserWarning, match="psd_wrap"):
        assert cp.Constant(_dense_psd_with_large_off_diagonals(6)).is_psd()


def test_is_psd_arpack_no_convergence(monkeypatch) -> None:
    """Test that ArpackNoConvergence is augmented with a helpful message."""
    n = 10

    def mock_eigsh(*args, **kwargs):
        raise sparla.ArpackNoConvergence("mock failure", np.array([]), np.array([[]]))

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)

    # Construct a sparse matrix that bypasses the Gershgorin and diagonal fast paths
    # so it reaches the ARPACK code path. Dense matrices take the pivoted
    # Cholesky path and never reach ARPACK.
    A = sp.csc_array(np.full((n, n), 0.5))
    A.setdiag(1.0)

    with pytest.raises(sparla.ArpackNoConvergence, match="CVXPY note"):
        cp.Constant(A).is_psd()


def test_is_psd_low_rank_matrix() -> None:
    """Regression test for inaccurate shift-invert eigenpairs (issue 2031)."""
    rng = np.random.RandomState(42)
    x = rng.randn(200, 2)
    A = x @ x.T

    assert cp.Constant(sp.csc_array(A)).is_psd()


def test_is_psd_inaccurate_eigenpair_fallback(monkeypatch) -> None:
    """Reject unreliable sparse eigenpairs after retrying ARPACK."""
    n = 10
    A = np.full((n, n), 0.5)
    np.fill_diagonal(A, 1.0)

    def mock_eigsh(matrix, *args, **kwargs):
        eigvec = np.ones((matrix.shape[0], 1)) / np.sqrt(matrix.shape[0])
        return np.array([-1.0]), eigvec

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)

    with pytest.raises(sparla.ArpackNoConvergence) as exc:
        cp.Constant(sp.csc_array(A)).is_psd()
    assert "ARPACK returned an inaccurate eigenpair" in str(exc.value)
    assert "CVXPY note" in str(exc.value)


def test_is_psd_negative_diagonal_witness(monkeypatch) -> None:
    """A diagonal entry below -tol decides the sparse PSD check before ARPACK."""
    def fail_eigsh(*args, **kwargs):
        raise AssertionError("eigsh should not be called")

    monkeypatch.setattr(sparla, 'eigsh', fail_eigsh)

    A = sp.block_diag([sp.csc_array([[-2e-10]]), 1e6 * np.ones((3, 3))], format='csc')
    assert not cp.Constant(A).is_psd()


def test_is_psd_diagonal_within_tolerance() -> None:
    """A diagonal entry within -tol must not be treated as a negative witness."""
    A = sp.block_diag([sp.csc_array([[-5e-11]]), np.ones((3, 3))], format='csc')
    assert cp.Constant(A).is_psd()


def _not_psd_with_nonnegative_diagonal(n=10):
    # Eigenvalues are 0.5 * (n - 1) (eigenvector: all ones) and -0.5, and the zero
    # diagonal skips the diagonal-witness fast path. The returned witness mixes a
    # -0.5 eigenvector with some of the positive eigenvector, so it is not an
    # eigenvector but its Ritz value is still clearly negative.
    A = np.full((n, n), 0.5)
    np.fill_diagonal(A, 0.0)
    negative_dir = np.zeros((n, 1))
    negative_dir[0], negative_dir[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    positive_dir = np.ones((n, 1)) / np.sqrt(n)
    v = negative_dir + 0.3 * positive_dir
    ritz = (v.T @ A @ v).item() / (v.T @ v).item()
    assert ritz < -0.05
    assert np.linalg.norm(A @ v - ritz * v) > 0.1  # not an eigenvector
    return sp.csc_array(A), v


@pytest.mark.parametrize("witness_on_retry", [False, True])
def test_is_psd_ritz_witness_from_inaccurate_eigenpair(monkeypatch, witness_on_retry) -> None:
    """A negative Ritz value certifies non-PSD even if the eigenpair is inaccurate."""
    A, v = _not_psd_with_nonnegative_diagonal()

    calls = []

    def mock_eigsh(matrix, *args, **kwargs):
        calls.append(kwargs.get('sigma'))
        if witness_on_retry and len(calls) == 1:
            return np.array([5.0]), np.ones_like(v) / np.sqrt(A.shape[0])
        return np.array([5.0]), v

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)
    assert not cp.Constant(A).is_psd()
    assert calls == ([-s.EIGVAL_TOL, None] if witness_on_retry else [-s.EIGVAL_TOL])


def test_is_psd_ritz_witness_from_arpack_no_convergence(monkeypatch) -> None:
    """Every partial vector from a non-converged ARPACK run is checked as a witness."""
    A, v = _not_psd_with_nonnegative_diagonal()
    positive = np.ones_like(v) / np.sqrt(A.shape[0])  # Ritz value 0.5 * (n - 1)

    def mock_eigsh(matrix, *args, **kwargs):
        raise sparla.ArpackNoConvergence(
            "no convergence", np.array([5.0, 5.0]), np.hstack([positive, v])
        )

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)
    assert not cp.Constant(A).is_psd()


def test_is_psd_arpack_no_convergence_without_witness(monkeypatch) -> None:
    """Without a negative Ritz value, non-convergence still raises with the CVXPY note."""
    A, _ = _not_psd_with_nonnegative_diagonal()
    n = A.shape[0]
    positive = np.ones((n, 1)) / np.sqrt(n)
    zero = np.zeros((n, 1))

    def mock_eigsh(matrix, *args, **kwargs):
        raise sparla.ArpackNoConvergence(
            "no convergence", np.array([5.0, 5.0]), np.hstack([positive, zero])
        )

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)
    with pytest.raises(sparla.ArpackNoConvergence, match="CVXPY note"):
        cp.Constant(A).is_psd()


def test_is_psd_ritz_witness_complex_hermitian(monkeypatch) -> None:
    """The Ritz value uses the conjugate inner product for complex Hermitian input."""
    A = sp.csc_array([[0, 1j], [-1j, 0]])  # eigenvalues -1 and 1
    v = np.array([[1], [2j]]) / np.sqrt(5)  # not an eigenvector; v^H A v = -0.8

    def mock_eigsh(matrix, *args, **kwargs):
        return np.array([5.0]), v

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)
    assert not cp.Constant(A).is_psd()


def test_is_psd_ritz_witness_roundoff_margin(monkeypatch) -> None:
    """A quotient inside the rounding margin must not certify non-PSD."""
    tol = s.EIGVAL_TOL
    off_diag = np.nextafter(1.0 + tol, np.inf)
    A = sp.csc_array([[1.0, off_diag], [off_diag, 1.0]])
    v = np.array([[1.0], [-1.0]]) / np.sqrt(2)
    # The quotient is below -tol by less than the rounding margin.
    assert np.vdot(v, A @ v).real / np.vdot(v, v).real < -tol

    def mock_eigsh(matrix, *args, **kwargs):
        return np.array([5.0]), v

    monkeypatch.setattr(sparla, 'eigsh', mock_eigsh)
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


def test_sparse_expression_values_are_sparse_arrays():
    lhs = cp.Constant(sp.csc_matrix(np.eye(2)))
    rhs = cp.Constant(np.ones((2, 2)))

    multiply_value = cp.multiply(lhs, rhs).value
    assert sp.issparse(multiply_value)
    assert isinstance(multiply_value, sp.sparray)
    assert not isinstance(multiply_value, sp.spmatrix)

    matmul_value = (lhs @ lhs).value
    assert sp.issparse(matmul_value)
    assert isinstance(matmul_value, sp.sparray)
    assert not isinstance(matmul_value, sp.spmatrix)


def test_mutating_source_array_does_not_change_constant():
    # Mutating the array a Constant was built from must not desynchronize
    # the constant's value from its cached sign attributes (issue: stale cache).
    a = np.array([1.0, 2.0])
    c = cp.Constant(a)
    assert c.is_nonneg()
    a[0] = -5.0
    assert np.allclose(c.value, [1.0, 2.0])
    assert c.is_nonneg()

    m = sp.csc_array(np.array([[1.0, 0.0], [0.0, 2.0]]))
    cs = cp.Constant(m)
    assert cs.is_nonneg()
    m.data[0] = -3.0
    assert np.allclose(cs.value.toarray(), [[1.0, 0.0], [0.0, 2.0]])
    assert cs.is_nonneg()


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
