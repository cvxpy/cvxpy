"""
qdldl on rank-deficient sparse symmetric matrices: failure-mode survey.

Two layers are probed:

  (A) qdldl directly:        qdldl.Solver(triu(A), upper=True).factors()
  (B) cvxpy.utilities.linalg.sparse_cholesky, which wraps qdldl with a
      prepass for zero-diagonal / all-zero rows and a sign check on D.

Findings (qdldl 0.1.9.post1):
  - qdldl raises hard errors in three structural rank-deficient regimes:
      * "Matrix not properly upper-triangular" whenever ANY diagonal entry
        is structurally zero (elimination-tree analysis fails).
      * "Input matrix is not quasi-definite" whenever a numerical pivot
        becomes exactly zero before qdldl has decided the factorization is
        OK -- but the trigger is order-dependent and inconsistent across
        rank-deficient inputs.
  - When qdldl factor() *does* succeed on a singular matrix, solver.solve(b)
    silently returns finite garbage that does not satisfy A x = b.
  - cvxpy's sparse_cholesky inherits all of these. Its prepass strips
    zero-diagonal rows that are also all-zero, but the reduced submatrix
    can still hit qdldl's "not quasi-definite" path -- so legitimate PSD
    rank-deficient inputs (e.g. v v^T with v dense and nonzero) currently
    fail with FACTORIZATION_FAILED.
"""
import importlib.util
import sys
import types

# --- Stub cvxpy.settings so we can import sparse_cholesky without the full
#     package init (which currently has a build/install issue in this env).
_settings = types.ModuleType("cvxpy.settings")
_settings.CHOL_SYM_TOL = 1e-9
_settings.CHOL_ZERO_PIVOT_TOL = 1e-12
sys.modules.setdefault("cvxpy", types.ModuleType("cvxpy"))
sys.modules["cvxpy.settings"] = _settings

_cvxtypes = types.ModuleType("cvxpy.expressions.cvxtypes")
class _NotMatch:
    def __init__(self, *a, **k): pass
_cvxtypes.expression = lambda: _NotMatch
_cvxtypes.constant = lambda: _NotMatch
sys.modules["cvxpy.expressions"] = types.ModuleType("cvxpy.expressions")
sys.modules["cvxpy.expressions.cvxtypes"] = _cvxtypes

_spec = importlib.util.spec_from_file_location(
    "_linalg", "/home/user/cvxpy/cvxpy/utilities/linalg.py")
_linalg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_linalg)
sparse_cholesky = _linalg.sparse_cholesky

import numpy as np
import qdldl
import scipy.linalg as sla
import scipy.sparse as sp

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def reconstruct(L, D, p, n):
    """qdldl returns: A[p, :][:, p] = (I + L) diag(D) (I + L)^T"""
    I_plus_L = sp.eye_array(n, format='csc') + sp.csc_array(L)
    A_perm = (I_plus_L @ sp.diags_array(D) @ I_plus_L.T).toarray()
    pinv = np.argsort(p)
    return A_perm[np.ix_(pinv, pinv)]


def probe(label, A_dense):
    n = A_dense.shape[0]
    A_sym = (A_dense + A_dense.T) / 2
    rank = np.linalg.matrix_rank(A_sym)
    eigs = np.sort(np.linalg.eigvalsh(A_sym))
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"  shape={A_sym.shape}, rank={rank}, eigs={np.round(eigs,4)}")

    A_csc = sp.csc_array(A_sym)
    A_upper = sp.triu(A_csc, format='csc')

    # --- (A) raw qdldl -----------------------------------------------------
    print("  qdldl direct:")
    try:
        solver = qdldl.Solver(A_upper, upper=True)
        L, D, p = solver.factors()
        recon = reconstruct(L, D, p, n)
        err = np.linalg.norm(A_sym - recon)
        print(f"    OK  D={D},  ||A - LDL^T||={err:.2e}")
        # Check solve quality on singular system.
        rng = np.random.default_rng(0)
        b = rng.standard_normal(n)
        x = solver.solve(b)
        res = np.linalg.norm(A_sym @ x - b)
        finite = np.all(np.isfinite(x))
        print(f"    solve: ||Ax - b||={res:.2e}, x finite? {finite}")
    except Exception as e:
        print(f"    RAISED  {type(e).__name__}: {e}")

    # --- (B) cvxpy.utilities.linalg.sparse_cholesky ------------------------
    print("  cvxpy sparse_cholesky:")
    try:
        sign, L_chol, p = sparse_cholesky(A_csc)
        L_dense = L_chol.toarray() if sp.issparse(L_chol) else L_chol
        Lp = L_dense[p, :]
        err = np.linalg.norm(A_sym - sign * (Lp @ Lp.T))
        print(f"    OK  sign={sign}, L shape={L_dense.shape}, "
              f"||A - s*LL^T||={err:.2e}")
    except Exception as e:
        print(f"    RAISED  {type(e).__name__}: {e}")

    # --- (C) dense scipy.linalg.ldl, for reference -------------------------
    Ld, Dd, _ = sla.ldl(A_sym)
    err_d = np.linalg.norm(A_sym - Ld @ Dd @ Ld.T)
    print(f"  dense scipy.linalg.ldl:  ||A - LDL^T||={err_d:.2e}, "
          f"diag(D)={np.diag(Dd)}")


# -- 1. Sweet spot: full-rank quasi-definite -----------------------------
A = np.array([[2., 1., 0.],
              [1., 3., 0.],
              [0., 0., -1.]])
probe("Full-rank quasi-definite (3x3) -- qdldl's sweet spot", A)

# -- 2. Rank-1 PSD vv^T with all-nonzero v -------------------------------
v = np.array([1., 2., 3., 4.])
probe("Rank-1 PSD vv^T (4x4) -- triggers 'not quasi-definite' on a PSD input",
      np.outer(v, v))

# -- 3. Rank-2 PSD UU^T (this is what existing cvxpy tests look like) ----
rng = np.random.default_rng(1)
U = rng.standard_normal((5, 2))
probe("Rank-2 PSD UU^T (5x5) random -- this is what existing tests cover",
      U @ U.T)

# -- 4. Zero on diagonal (PSD) -- elimination-tree failure ---------------
probe("PSD with zero leading diagonal entry (3x3)",
      np.array([[0., 0., 0.],
                [0., 4., 2.],
                [0., 2., 1.]]))

# -- 5. Rank-1 PSD vv^T with one zero in v -- elimination-tree failure ---
probe("Rank-1 PSD with zero in v=[1,0,1,1] (4x4)",
      np.outer([1., 0., 1., 1.], [1., 0., 1., 1.]))

# -- 6. Rank-deficient indefinite -- qdldl HAPPENS to succeed ------------
diag = np.array([3., -2., 0., 0., 1.5])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A6 = (Q @ np.diag(diag) @ Q.T)
probe("Rank-3 indefinite (5x5) -- qdldl produces mixed-sign D and 'succeeds'",
      (A6 + A6.T) / 2)

# -- 7. Saddle-point KKT (rank-full but zero diagonal) -------------------
B = rng.standard_normal((3, 3))
probe("Saddle [[0,B],[B^T,0]] (6x6) -- well-conditioned, but qdldl fails",
      np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]]))

# -- 8. Rank-deficient quasi-definite ------------------------------------
probe("Rank-deficient quasi-definite (4x4)",
      np.array([[ 1.,  1.,  0.,  0.],
                [ 1.,  1.,  0.,  0.],
                [ 0.,  0., -1., -1.],
                [ 0.,  0., -1., -1.]]))

# -- 9. PSD with one all-zero row/col + nontrivial rest ------------------
v1 = np.array([1., 1., 0., 1.])
v2 = np.array([1., 0., 0., 1.])
probe("Rank-2 PSD with one all-zero row/col (4x4) -- prepass reduces, "
      "but reduced 3x3 still trips qdldl",
      np.outer(v1, v1) + np.outer(v2, v2))

# -- 10. Near-zero pivot -------------------------------------------------
v = np.array([1., 2., 3.])
probe("vv^T + 1e-14 I (3x3) -- numerical near-zero pivot",
      np.outer(v, v) + 1e-14 * np.eye(3))
