"""Minimal smoke test for the faer LDL wrapper."""
import numpy as np
import scipy.sparse as sp

import faer_ldl


def factor_solve(A_dense, b):
    """Pass upper triangle to the wrapper, return the solve x."""
    A = sp.csc_array(np.triu(A_dense))
    n = A.shape[0]
    ldl = faer_ldl.SparseLDL(
        n,
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data.astype(np.float64),
    )
    x = ldl.solve(np.asarray(b, dtype=np.float64))
    return ldl, x


def report(label, A_dense, expect_singular):
    n = A_dense.shape[0]
    rank = np.linalg.matrix_rank(A_dense)
    print(f"\n--- {label}  (n={n}, rank={rank}) ---")
    rng = np.random.default_rng(0)
    if expect_singular:
        x_true = rng.standard_normal(n)
        b = A_dense @ x_true
    else:
        b = rng.standard_normal(n)
    try:
        ldl, x = factor_solve(A_dense, b)
        res = np.linalg.norm(A_dense @ x - b)
        print(f"  factor OK, nnz_l={ldl.nnz_l}")
        print(f"  ||Ax - b|| = {res:.3e}, x finite? {np.all(np.isfinite(x))}")
    except Exception as e:
        print(f"  RAISED  {type(e).__name__}: {e}")


# Sweet spot
A = np.array([[2., 1., 0.], [1., 3., 0.], [0., 0., -1.]])
report("Full-rank quasi-definite (3x3)", A, expect_singular=False)

# Random full-rank symmetric
rng = np.random.default_rng(42)
M = rng.standard_normal((6, 6))
A = (M + M.T) / 2 + 4 * np.eye(6)
report("Full-rank dense symmetric (6x6)", A, expect_singular=False)

# Rank-deficient PSD
v = np.array([1., 2., 3., 4.])
report("Rank-1 PSD vv^T (4x4)", np.outer(v, v), expect_singular=True)

# Rank-2 PSD
U = rng.standard_normal((5, 2))
report("Rank-2 PSD UU^T (5x5)", U @ U.T, expect_singular=True)

# Saddle point (full rank, zero diagonal)
B = rng.standard_normal((3, 3))
report("Saddle [[0,B],[B^T,0]] (6x6)",
       np.block([[np.zeros((3,3)), B],[B.T, np.zeros((3,3))]]),
       expect_singular=False)

# Indefinite full-rank
diag = np.array([3., -2., 1.5, -1., 2.])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A = Q @ np.diag(diag) @ Q.T
A = (A + A.T) / 2
report("Indefinite full-rank (5x5)", A, expect_singular=False)
