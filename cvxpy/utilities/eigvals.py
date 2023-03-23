import numpy as np
import scipy.sparse as spar
import scipy.sparse.linalg as sparla


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
    A : Union[np.ndarray, spar.spmatrx]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.

    tol : float
        Nonnegative. Something very small, like 1e-10.
    """

    if gershgorin_psd_check(A, tol):
        return True

    def SA_eigsh(sigma):
        return sparla.eigsh(A, k=1, sigma=sigma, which='SA', return_eigenvectors=False)
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

    ev = np.NaN
    try:
        ev = SA_eigsh(-tol)  # might return np.NaN, or raise exception
    finally:
        if np.isnan(ev).all():
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
    A : Union[np.ndarray, spar.spmatrx]
        Symmetric (or Hermitian) NumPy ndarray or SciPy sparse matrix.

    tol : float
        Nonnegative. Something very small, like 1e-10.

    Returns
    -------
    True if A is PSD according to the Gershgorin Circle Theorem.
    Otherwise, return False.
    """
    if isinstance(A, spar.spmatrix):
        diag = A.diagonal()
        if np.any(diag < -tol):
            return False
        A_shift = A - spar.diags(diag)
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
