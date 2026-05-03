"""
Run the same rank-deficient battery against python-mumps to see what
MUMPS does where qdldl fails, and whether the API exposes the bits we
need (rank, null space, solve correctness).

Probed signals from MUMPS:
  - factor() success/failure
  - factor_stats.delayed_pivots / tiny_pivots / offdiag_pivots (2x2 blocks)
  - mumps_instance.infog: INFOG(12) = #negative pivots,
                          INFOG(28) = #null pivots detected
  - mumps.nullspace(A, symmetric=True): returns basis of null(A)
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

import mumps

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def probe(label, A_dense):
    n = A_dense.shape[0]
    A_sym = (A_dense + A_dense.T) / 2
    A_csc = sp.csc_array(A_sym)
    rank = np.linalg.matrix_rank(A_sym)
    eigs = np.sort(np.linalg.eigvalsh(A_sym))
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"  shape={A_sym.shape}, rank={rank}, eigs={np.round(eigs, 4)}")

    # -- MUMPS factor --------------------------------------------------------
    ctx = mumps.Context(verbose=False)
    try:
        ctx.set_matrix(A_csc, symmetric=True)
    except Exception as e:
        print(f"  set_matrix RAISED  {type(e).__name__}: {e}")
        return
    # Turn on null-pivot detection so factor() doesn't bail on rank deficiency.
    # ICNTL(24)=1: detect & report null pivots; CNTL(3): threshold.
    ctx.mumps_instance.icntl[24] = 1
    ctx.mumps_instance.cntl[3] = 1e-10
    try:
        ctx.factor(pivot_tol=0.01)
    except Exception as e:
        print(f"  factor RAISED      {type(e).__name__}: {e}")
        return

    fs = ctx.factor_stats
    infog = ctx.mumps_instance.infog
    # INFOG is 1-indexed; INFOG[12] = #negative pivots, INFOG[28] = #null pivots
    n_negative_pivots = int(infog[11])
    n_null_pivots = int(infog[27])
    print(f"  factor OK")
    print(f"    delayed_pivots={fs.delayed_pivots}, "
          f"offdiag_pivots(2x2)={fs.offdiag_pivots}, "
          f"tiny_pivots={fs.tiny_pivots}")
    print(f"    INFOG(12) #neg pivots = {n_negative_pivots}, "
          f"INFOG(28) #null pivots = {n_null_pivots}")
    print(f"    => MUMPS-implied rank = {n - n_null_pivots} "
          f"(numpy rank = {rank})")

    # -- MUMPS solve on a consistent and an inconsistent rhs ----------------
    rng = np.random.default_rng(0)
    if rank < n:
        # Build a rhs in range(A): b = A @ x_true
        x_true = rng.standard_normal(n)
        b_consistent = A_sym @ x_true
        try:
            x = ctx.solve(b_consistent)
            res = np.linalg.norm(A_sym @ x - b_consistent)
            print(f"    solve(consistent rhs):   ||Ax - b||={res:.2e}, "
                  f"finite? {np.all(np.isfinite(x))}")
        except Exception as e:
            print(f"    solve(consistent rhs) RAISED  {type(e).__name__}: {e}")
        b_random = rng.standard_normal(n)
        try:
            x = ctx.solve(b_random)
            res = np.linalg.norm(A_sym @ x - b_random)
            print(f"    solve(random rhs):       ||Ax - b||={res:.2e}, "
                  f"finite? {np.all(np.isfinite(x))}")
        except Exception as e:
            print(f"    solve(random rhs) RAISED  {type(e).__name__}: {e}")
    else:
        b = rng.standard_normal(n)
        x = ctx.solve(b)
        res = np.linalg.norm(A_sym @ x - b)
        print(f"    solve: ||Ax - b||={res:.2e}, finite? {np.all(np.isfinite(x))}")

    # -- mumps.nullspace -----------------------------------------------------
    try:
        # nullspace expects upper triangle for symmetric input
        A_upper = sp.triu(A_csc).tocsc()
        N = mumps.nullspace(A_upper, symmetric=True, pivot_threshold=1e-10)
        print(f"  nullspace: shape={N.shape}")
        if N.size > 0:
            AN_norm = np.linalg.norm(A_sym @ N, ord='fro')
            # Also check independence (rank of N)
            N_rank = np.linalg.matrix_rank(N) if N.size else 0
            print(f"    ||A @ N||_F = {AN_norm:.2e}, rank(N) = {N_rank}, "
                  f"expected nullity = {n - rank}")
    except Exception as e:
        print(f"  nullspace RAISED  {type(e).__name__}: {e}")


# Same battery as the qdldl script.
A = np.array([[2., 1., 0.],
              [1., 3., 0.],
              [0., 0., -1.]])
probe("Full-rank quasi-definite (3x3)", A)

v = np.array([1., 2., 3., 4.])
probe("Rank-1 PSD vv^T (4x4) -- qdldl FAILS, scipy dense ldl OK", np.outer(v, v))

rng = np.random.default_rng(1)
U = rng.standard_normal((5, 2))
probe("Rank-2 PSD UU^T (5x5)", U @ U.T)

probe("PSD with zero leading diagonal entry (3x3) -- qdldl FAILS",
      np.array([[0., 0., 0.], [0., 4., 2.], [0., 2., 1.]]))

probe("Rank-1 PSD with zero in v=[1,0,1,1] (4x4) -- qdldl FAILS",
      np.outer([1., 0., 1., 1.], [1., 0., 1., 1.]))

diag = np.array([3., -2., 0., 0., 1.5])
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A6 = (Q @ np.diag(diag) @ Q.T)
probe("Rank-3 indefinite (5x5)", (A6 + A6.T) / 2)

B = rng.standard_normal((3, 3))
probe("Saddle [[0,B],[B^T,0]] (6x6) -- qdldl FAILS, full rank",
      np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]]))

probe("Rank-deficient quasi-definite (4x4) -- qdldl FAILS",
      np.array([[ 1.,  1.,  0.,  0.],
                [ 1.,  1.,  0.,  0.],
                [ 0.,  0., -1., -1.],
                [ 0.,  0., -1., -1.]]))

v1 = np.array([1., 1., 0., 1.])
v2 = np.array([1., 0., 0., 1.])
probe("Rank-2 PSD with one all-zero row/col (4x4) -- qdldl FAILS",
      np.outer(v1, v1) + np.outer(v2, v2))

v = np.array([1., 2., 3.])
probe("vv^T + 1e-14 I (3x3) -- numerical near-zero pivot",
      np.outer(v, v) + 1e-14 * np.eye(3))

# Bigger sanity check with random rank-deficient indefinite
rng2 = np.random.default_rng(7)
n, k = 30, 15
M = rng2.standard_normal((n, k))
S = np.diag(rng2.choice([-1., 1.], size=k))
A_big = M @ S @ M.T
probe("Random indefinite rank-15 (30x30)", A_big)
