import numpy as np
import qdldl
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_array

import cvxpy.settings as settings


def orth(V, tol=1e-12):
    """Return a matrix whose columns are an orthonormal basis for range(V)"""
    Q, R, p = la.qr(V, mode='economic', pivoting=True)
    # ^ V[:, p] == Q @ R.
    rank = np.count_nonzero(np.sum(np.abs(R) > tol, axis=1))
    Q = Q[:, :rank].reshape((V.shape[0], rank))  # ensure 2-dimensional
    return Q


def onb_for_orthogonal_complement(V):
    """
    Let U = the orthogonal complement of range(V).

    This function returns an array Q whose columns are
    an orthonormal basis for U. It requires that dim(U) > 0.
    """
    n = V.shape[0]
    Q1 = orth(V)
    rank = Q1.shape[1]
    assert n > rank
    if np.iscomplexobj(V):
        P = np.eye(n) - Q1 @ Q1.conj().T
    else:
        P = np.eye(n) - Q1 @ Q1.T
    Q2 = orth(P)
    return Q2


def is_diagonal(A):
    if sp.issparse(A):
        off_diagonal_elements = A - sp.diags_array(A.diagonal())
        off_diagonal_elements = off_diagonal_elements.toarray()
    elif isinstance(A, np.ndarray):
        off_diagonal_elements = A - np.diag(np.diag(A))
    else:
        raise ValueError("Unsupported matrix type.")

    return np.allclose(off_diagonal_elements, 0)


def is_psd_within_tol(A, tol):
    """
    Return True if we can certify that A is PSD (up to tolerance "tol").

    First we check if A is PSD according to the Gershgorin Circle Theorem.

    If Gershgorin is inconclusive, then we use an iterative method (from ARPACK,
    as called through SciPy) to estimate extremal eigenvalues of certain shifted
    versions of A. The shifts are chosen so that the signs of those eigenvalues
    tell us the signs of the eigenvalues of A.

    If there are numerical issues then it's possible that this function returns
    False even when A is PSD. If you know that you're in that situation, then
    you should replace A by

        A = cvxpy.atoms.affine.wraps.psd_wrap(A).

    Parameters
    ----------
    A : Union[np.ndarray, sp.sparray]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse array.

    tol : float
        Nonnegative. Something very small, like 1e-10.
    """

    if gershgorin_psd_check(A, tol):
        return True

    if is_diagonal(A):
        if isinstance(A, csc_array):
            return np.all(A.data >= -tol)
        else:
            min_diag_entry = np.min(np.diag(A))
            return min_diag_entry >= -tol

    def SA_eigsh(sigma):

        # Check for default_rng in np.random module (new API)
        if hasattr(np.random, 'default_rng'):
            g = np.random.default_rng(123)
        else:  # fallback to legacy RandomState
            g = np.random.RandomState(123)

        n = A.shape[0]
        v0 = g.normal(loc=0.0, scale=1.0, size=n)

        return sparla.eigsh(A, k=1, sigma=sigma, which='SA', v0=v0,
                            return_eigenvectors=False)
        # Returns the eigenvalue w[i] of A where 1/(w[i] - sigma) is minimized.
        #
        # If A - sigma*I is PSD, then w[i] should be equal to the largest
        # eigenvalue of A.
        #
        # If A - sigma*I is not PSD, then w[i] should be the largest eigenvalue
        # of A where w[i] - sigma < 0.
        #
        # We should only call this function with sigma < 0. In this case, if
        # A - sigma*I is not PSD then A is not PSD, and w[i] < -abs(sigma) is
        # a negative eigenvalue of A. If A - sigma*I is PSD, then we obviously
        # have that the smallest eigenvalue of A is >= sigma.

    try:
        ev = SA_eigsh(-tol)  # might return np.NaN, or raise exception
    except sparla.ArpackNoConvergence as e:
        # This is a numerical issue. We can't certify that A is PSD.

        message = """
        CVXPY note: This failure was encountered while trying to certify
        that a matrix is positive semi-definite (see [1] for a definition).
        In rare cases, this method fails for numerical reasons even when the matrix is
        positive semi-definite. If you know that you're in that situation, you can
        replace the matrix A by cvxpy.psd_wrap(A).

        [1] https://en.wikipedia.org/wiki/Definite_matrix
        """

        error_with_note = f"{str(e)}\n\n{message}"

        raise sparla.ArpackNoConvergence(error_with_note, e.eigenvalues, e.eigenvectors)

    if np.isnan(ev).any():
        # will be NaN if A has an eigenvalue which is exactly -tol
        # (We might also hit this code block for other reasons.)
        temp = tol - np.finfo(A.dtype).eps
        ev = SA_eigsh(-temp)

    return np.all(ev >= -tol)


def gershgorin_psd_check(A, tol):
    """
    Use the Gershgorin Circle Theorem

        https://en.wikipedia.org/wiki/Gershgorin_circle_theorem

    As a sufficient condition for A being PSD with tolerance "tol".

    The computational complexity of this function is O(nnz(A)).

    Parameters
    ----------
    A : Union[np.ndarray, sp.sparray]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse array.

    tol : float
        Nonnegative. Something very small, like 1e-10.

    Returns
    -------
    True if A is PSD according to the Gershgorin Circle Theorem.
    Otherwise, return False.
    """
    if sp.issparse(A):
        diag = A.diagonal()
        if np.any(diag < -tol):
            return False
        A_shift = A - sp.diags_array(diag)
        A_shift = np.abs(A_shift)
        radii = np.array(A_shift.sum(axis=0)).ravel()
        return np.all(diag - radii >= -tol)
    elif isinstance(A, np.ndarray):
        diag = np.diag(A)
        if np.any(diag < -tol):
            return False
        A_shift = A - np.diag(diag)
        A_shift = np.abs(A_shift)
        radii = A_shift.sum(axis=0)
        return np.all(diag - radii >= -tol)
    else:
        raise ValueError()


class SparseCholeskyMessages:

    ASYMMETRIC = 'Input matrix is not symmetric to within provided tolerance.'
    INDEFINITE = 'Input matrix is neither positive nor negative definite.'
    FACTORIZATION_FAILED = 'Cholesky factorization failed.'
    NOT_CONST = 'The only allowed Expression inputs are Constant objects.'
    NOT_SPARSE = 'Non-Expression inputs must be SciPy sparse matrices.'
    NOT_REAL = 'Input matrix must be real.'


def sparse_cholesky(A, sym_tol=settings.CHOL_SYM_TOL, assume_psd=False):
    """
    The input matrix A must be real and symmetric. If A is positive (semi)definite
    then QDLDL will be used to compute its sparse LDL factorization with fill-reducing
    ordering, which is then converted to Cholesky form with zero-pivot columns dropped.
    If A is negative (semi)definite, then the analogous operation will be applied to -A.

    If factorization succeeds, then we return (sign, L, p) where sign is +1.0
    for positive (semi)definite or -1.0 for negative (semi)definite, L is an
    n-by-k matrix in CSR-format (where k = rank(A)), and p is a permutation
    vector so that (L[p, :]) @ (L[p, :]).T == sign * A within numerical precision.

    We raise a ValueError if QDLDL's factorization reveals indefiniteness or if
    we certify indefiniteness before calling QDLDL. While checking for
    indefiniteness, we also check that
     ||A - A'||_Fro / sqrt(n) <= sym_tol, where n is the order of the matrix.
    """
    import cvxpy.expressions.cvxtypes as cvxtypes

    if isinstance(A, cvxtypes.expression()):
        if not isinstance(A, cvxtypes.constant()):
            raise ValueError(SparseCholeskyMessages.NOT_CONST)
        A = A.value

    if not sp.issparse(A):
        raise ValueError(SparseCholeskyMessages.NOT_SPARSE)
    if np.iscomplexobj(A):
        raise ValueError(SparseCholeskyMessages.NOT_REAL)

    if not assume_psd:
        # check that we're symmetric
        symdiff = A - A.T
        sz = symdiff.data.size
        if sz > 0 and la.norm(symdiff.data) > sym_tol * (sz**0.5):
            raise ValueError(SparseCholeskyMessages.ASYMMETRIC)
        # check a necessary condition for positive/negative semidefiniteness.
        d = A.diagonal()
        maybe_psd = np.all(d >= 0)
        maybe_nsd = np.all(d <= 0)
        if not (maybe_psd or maybe_nsd):
            raise ValueError(SparseCholeskyMessages.INDEFINITE)

    n = A.shape[0]

    # QDLDL cannot handle rows/columns that are entirely zero (it fails with
    # "Matrix not properly upper-triangular"). Detect zero-diagonal entries
    # and reduce to the nonzero submatrix. For a PSD/NSD matrix, zero diagonal
    # implies all-zero row/column; we validate this before reducing.
    diag = A.diagonal()
    nonzero_idx = np.flatnonzero(diag)
    if nonzero_idx.size < n:
        # Zero-diagonal rows must be all-zero for a PSD/NSD matrix. Otherwise
        # the matrix has a zero on the diagonal paired with off-diagonal
        # entries, which is indefinite (e.g. [[1, 2], [2, 0]] or the saddle
        # [[0, B], [B^T, 0]]).
        zero_idx = np.setdiff1d(np.arange(n), nonzero_idx)
        if A[zero_idx, :].nnz > 0:
            raise ValueError(SparseCholeskyMessages.INDEFINITE)
    if nonzero_idx.size == 0:
        # All-zero matrix (validated above): return n-by-0 factor.
        return 1.0, sp.csr_array((n, 0)), np.arange(n)
    if nonzero_idx.size < n:
        # Reduce to the nonzero submatrix, factorize, then expand back.
        A_sub = A[np.ix_(nonzero_idx, nonzero_idx)]
        sign, L_sub, p_sub = sparse_cholesky(A_sub, sym_tol, assume_psd=True)
        # Expand L_sub back to L (n-by-k) by inserting zero rows.
        k = L_sub.shape[1]
        L_expanded = sp.lil_array((n, k))
        L_expanded[nonzero_idx, :] = L_sub[p_sub, :]
        L_expanded = sp.csr_array(L_expanded)
        return sign, L_expanded, np.arange(n)

    # QDLDL expects upper triangular CSC format
    A_upper = sp.triu(A, format='csc')

    tol = settings.CHOL_ZERO_PIVOT_TOL

    try:
        solver = qdldl.Solver(A_upper, upper=True)
        L_unit, D, p = solver.factors()

        # QDLDL returns: A[p,:][:,p] = (I + L) @ diag(D) @ (I + L).T
        # Determine sign from D: all D >= 0 means PSD, all D <= 0 means NSD.
        scale = np.max(np.abs(D))
        if scale == 0:
            # All pivots are zero — zero matrix.
            return 1.0, sp.csr_array((n, 0)), np.arange(n)

        is_psd = np.all(D >= -tol * scale)
        is_nsd = np.all(D <= tol * scale)
        if not (is_psd or is_nsd):
            raise ValueError(SparseCholeskyMessages.INDEFINITE)

        sign = 1.0 if is_psd else -1.0

        # Build L_chol = (I + L_unit) @ diag(sqrt(|D|)), keeping only columns
        # where |D| is significantly nonzero to get an n-by-rank(A) factor.
        abs_D = np.abs(D)
        mask = abs_D > tol * scale
        sqrt_D_vals = np.sqrt(np.maximum(abs_D[mask], 0))
        I_plus_L = sp.eye_array(n, format='csr') + sp.csr_array(L_unit)
        L_chol = I_plus_L[:, mask] @ sp.diags_array(sqrt_D_vals, format='csr')
        L_chol = sp.csr_array(L_chol)

        # QDLDL's permutation p satisfies A[p,:][:,p] = L_full @ L_full.T.
        # The caller expects L_out[p_out, :] @ L_out[p_out, :].T == sign * A,
        # so we need p_out = argsort(p) (the inverse permutation).
        p_inv = np.argsort(p)

        return sign, L_chol, p_inv

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"{SparseCholeskyMessages.FACTORIZATION_FAILED}: {e}"
        ) from e
