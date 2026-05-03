"""End-to-end check: build M1, M2 from faer_ldl factors and verify
P = M1 M1^T - M2 M2^T (the contract decomp_quad needs).

This mirrors what cvxpy.atoms.quad_form.decomp_quad does after a dense
scipy.linalg.ldl call -- extract 2x2 blocks via eigh, split columns by
sign of the resulting diagonal.
"""
import numpy as np
import scipy.sparse as sp

import faer_ldl

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def decomp_quad_via_faer(P_dense, cond=1e-10):
    n = P_dense.shape[0]
    P_sym = (P_dense + P_dense.T) / 2
    P_upper = sp.csc_array(np.triu(P_sym))
    ldl = faer_ldl.SparseLDL(
        n,
        P_upper.indptr.astype(np.int64),
        P_upper.indices.astype(np.int64),
        P_upper.data.astype(np.float64),
    )
    Li, Lj, Lv, Dd, Ds, perm = ldl.factor()
    L = sp.csc_array((Lv, Lj, Li), shape=(n, n)).toarray()
    I_plus_L = np.eye(n) + L  # the columns of (I+L) are what get scaled

    # Resolve 2x2 blocks via eigh, mirroring the dense decomp_quad logic.
    diag_vals = Dd.copy()
    bs = np.flatnonzero(Ds[:-1] != 0)  # block starts
    if bs.size:
        idx = bs[:, None] + np.arange(2)[None, :]
        # build (k, 2, 2) blocks
        blocks = np.zeros((len(bs), 2, 2))
        for k, i in enumerate(bs):
            blocks[k] = [[Dd[i], Ds[i]], [Ds[i], Dd[i + 1]]]
        eigvals, eigvecs = np.linalg.eigh(blocks)
        diag_vals[bs] = eigvals[:, 0]
        diag_vals[bs + 1] = eigvals[:, 1]
        # rotate the corresponding pair of columns of (I+L)
        Li_col = I_plus_L[:, bs].copy()
        Lp1_col = I_plus_L[:, bs + 1].copy()
        I_plus_L[:, bs] = Li_col * eigvecs[:, 0, 0] + Lp1_col * eigvecs[:, 1, 0]
        I_plus_L[:, bs + 1] = Li_col * eigvecs[:, 0, 1] + Lp1_col * eigvecs[:, 1, 1]

    scale = max(np.abs(diag_vals).max(), 0.0)
    if scale == 0:
        d_scaled = diag_vals
    else:
        d_scaled = diag_vals / scale
    maskp = d_scaled > cond
    maskn = d_scaled < -cond

    inv_perm = np.argsort(perm)
    # The factor lives in the permuted basis: P[perm, :][:, perm] = (I+L) D (I+L)^T
    # so M1, M2 in the original basis are obtained by undoing perm.
    cols_p = I_plus_L[:, maskp] * np.sqrt(d_scaled[maskp])
    cols_n = I_plus_L[:, maskn] * np.sqrt(-d_scaled[maskn])
    M1 = cols_p[inv_perm, :] if cols_p.size else np.zeros((n, 0))
    M2 = cols_n[inv_perm, :] if cols_n.size else np.zeros((n, 0))
    return scale, M1, M2


def check(label, P):
    P = (P + P.T) / 2
    rank = np.linalg.matrix_rank(P)
    scale, M1, M2 = decomp_quad_via_faer(P)
    P_back = scale * (M1 @ M1.T - M2 @ M2.T)
    err = np.linalg.norm(P - P_back)
    rel = err / max(np.linalg.norm(P), 1e-300)
    print(f"{label:<50s}  rank={rank}, M1={M1.shape}, M2={M2.shape}, "
          f"||P - s(M1 M1^T - M2 M2^T)||={err:.3e} (rel={rel:.2e})")


rng = np.random.default_rng(7)

# PSD rank-deficient
v = np.array([1., 2., 3., 4.])
check("Rank-1 PSD vv^T (4x4)", np.outer(v, v))

U = rng.standard_normal((6, 3))
check("Rank-3 PSD UU^T (6x6)", U @ U.T)

# Indefinite full-rank
diag = np.array([3., -2., 1.5, -1., 2.])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A = Q @ np.diag(diag) @ Q.T
check("Indefinite full-rank (5x5)", (A + A.T) / 2)

# Indefinite rank-deficient
diag = np.array([3., -2., 0., 0., 1.5])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A = Q @ np.diag(diag) @ Q.T
check("Indefinite rank-3 (5x5)", (A + A.T) / 2)

# Saddle point: indefinite, full-rank, all-zero diagonal -- the case both
# qdldl and cvxpy's sparse_cholesky get wrong today.
B = rng.standard_normal((3, 3))
A = np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]])
check("Saddle [[0,B],[B^T,0]] (6x6)", A)

# Bigger sparse-ish indefinite
n, k = 30, 10
M = rng.standard_normal((n, k))
S = np.diag(rng.choice([-1., 1.], size=k))
A = M @ S @ M.T
check(f"Indefinite rank-{k} ({n}x{n})", A)

# Diagonal PSD with a zero
check("PSD with zero leading diagonal entry (3x3)",
      np.array([[0., 0., 0.], [0., 4., 2.], [0., 2., 1.]]))
