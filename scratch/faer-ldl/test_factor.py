"""Verify faer_ldl.SparseLDL.factor() returns a valid L, D, perm.

We want to confirm that for symmetric A (we pass upper triangle):
    A[perm, :][:, perm] ≈ (I + L) D (I + L)^T
where L is unit lower triangular (diagonal implicit), D is block-diagonal
with 1×1 entries from D_diag and 2×2 blocks where D_subdiag[i] != 0.
"""
import numpy as np
import scipy.sparse as sp

import faer_ldl

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def reconstruct(L_indptr, L_indices, L_data, D_diag, D_subdiag, perm, n):
    """Build (I + L) D (I + L)^T from the returned components."""
    L = sp.csc_array((L_data, L_indices, L_indptr), shape=(n, n)).toarray()
    I_plus_L = np.eye(n) + L
    # Build D as a dense block diagonal matrix.
    D = np.diag(D_diag)
    for i in range(n - 1):
        if D_subdiag[i] != 0.0:
            D[i + 1, i] = D_subdiag[i]
            D[i, i + 1] = D_subdiag[i]
    return I_plus_L @ D @ I_plus_L.T, L, D


def probe(label, A_dense):
    n = A_dense.shape[0]
    A_sym = (A_dense + A_dense.T) / 2
    rank = np.linalg.matrix_rank(A_sym)
    eigs = np.sort(np.linalg.eigvalsh(A_sym))
    print(f"\n--- {label}  (n={n}, rank={rank}, eigs={np.round(eigs,3)}) ---")

    A_upper = sp.csc_array(np.triu(A_sym))
    try:
        ldl = faer_ldl.SparseLDL(
            n,
            A_upper.indptr.astype(np.int64),
            A_upper.indices.astype(np.int64),
            A_upper.data.astype(np.float64),
        )
    except Exception as e:
        print(f"  ctor RAISED  {type(e).__name__}: {e}")
        return

    Li, Lj, Lv, Dd, Ds, perm = ldl.factor()
    A_back, L, D = reconstruct(Li, Lj, Lv, Dd, Ds, perm, n)

    # Check (I + L) D (I + L)^T == A[perm, :][:, perm]
    A_perm = A_sym[perm][:, perm]
    err = np.linalg.norm(A_perm - A_back)
    print(f"  L nnz = {len(Lv)}, D_diag = {np.round(Dd, 3)}")
    if np.any(Ds != 0):
        which = np.flatnonzero(Ds)
        print(f"  D_subdiag nonzero at indices {which}: values {np.round(Ds[which], 3)}")
    print(f"  perm = {perm.tolist()}")
    print(f"  ||A[perm,:][:,perm] - (I+L) D (I+L)^T||_F = {err:.3e}")


probe("Full-rank quasi-definite (3x3)",
      np.array([[2., 1., 0.], [1., 3., 0.], [0., 0., -1.]]))

rng = np.random.default_rng(42)
M = rng.standard_normal((6, 6))
probe("Full-rank dense symmetric (6x6)",
      (M + M.T) / 2 + 4 * np.eye(6))

probe("Rank-1 PSD vv^T (4x4)",
      np.outer([1., 2., 3., 4.], [1., 2., 3., 4.]))

U = rng.standard_normal((5, 2))
probe("Rank-2 PSD UU^T (5x5)", U @ U.T)

B = rng.standard_normal((3, 3))
probe("Saddle [[0,B],[B^T,0]] (6x6)",
      np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]]))

diag = np.array([3., -2., 1.5, -1., 2.])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A = Q @ np.diag(diag) @ Q.T
probe("Indefinite full-rank (5x5)", (A + A.T) / 2)
