"""
Test how scipy.linalg.ldl handles low-rank (rank-deficient) matrices.

scipy.linalg.ldl computes the Bunch-Kaufman LDL^T factorization for
symmetric/Hermitian matrices: A = L D L^T (or A = L D L^H).
D is block-diagonal with 1x1 and 2x2 blocks.

Low-rank matrices are singular, so the factorization may produce
zero diagonal entries in D, signaling singularity.
"""
import numpy as np
from scipy.linalg import ldl

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def report(label, A, hermitian=False):
    print(f"\n{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print(f"A shape: {A.shape}")
    rank = np.linalg.matrix_rank(A)
    print(f"numpy rank: {rank} / {A.shape[0]}")
    eigs = np.linalg.eigvalsh(A) if not hermitian else np.linalg.eigvalsh(A)
    print(f"eigenvalues: {eigs}")

    try:
        L, D, perm = ldl(A, hermitian=hermitian)
    except Exception as e:
        print(f"ldl FAILED: {type(e).__name__}: {e}")
        return

    print(f"\nL (permuted rows; L[perm] is lower-triangular):")
    print(L)
    print(f"\nD (block diagonal, 1x1 / 2x2 blocks):")
    print(D)
    print(f"\nperm: {perm}")

    # Reconstruct: A = L @ D @ L.T  (or .conj().T for hermitian)
    LT = L.conj().T if hermitian else L.T
    A_reconstructed = L @ D @ LT
    err = np.linalg.norm(A - A_reconstructed)
    print(f"\n||A - L D L^T||_F = {err:.3e}")

    diag_D = np.diag(D)
    print(f"diag(D) = {diag_D}")
    n_zero = int(np.sum(np.abs(diag_D) < 1e-10))
    print(f"# near-zero diagonal entries in D: {n_zero}")


# ---------------------------------------------------------------
# 1. Full-rank PSD (sanity check)
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.standard_normal((5, 5))
A = X @ X.T + 0.1 * np.eye(5)
report("Full-rank SPD (5x5)", A)

# ---------------------------------------------------------------
# 2. Rank-1 symmetric PSD
# ---------------------------------------------------------------
v = np.array([1.0, 2.0, 3.0, 4.0])
A = np.outer(v, v)  # rank 1
report("Rank-1 PSD (4x4): v v^T", A)

# ---------------------------------------------------------------
# 3. Rank-2 PSD inside 5x5
# ---------------------------------------------------------------
U = rng.standard_normal((5, 2))
A = U @ U.T  # rank 2
report("Rank-2 PSD (5x5): U U^T with U 5x2", A)

# ---------------------------------------------------------------
# 4. Rank-deficient symmetric INDEFINITE (eigenvalues both signs + zero)
# ---------------------------------------------------------------
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
diag = np.array([3.0, -2.0, 1.5, 0.0, 0.0])  # rank 3, indefinite
A = Q @ np.diag(diag) @ Q.T
A = (A + A.T) / 2  # enforce symmetry
report("Rank-3 indefinite symmetric (5x5)", A)

# ---------------------------------------------------------------
# 5. All-zero matrix (rank 0)
# ---------------------------------------------------------------
A = np.zeros((4, 4))
report("Zero matrix (rank 0, 4x4)", A)

# ---------------------------------------------------------------
# 6. Rank-1 with leading zero (forces a pivot / 2x2 block)
# ---------------------------------------------------------------
v = np.array([0.0, 1.0, 1.0])
A = np.outer(v, v)
report("Rank-1 PSD with zero leading entry (3x3)", A)

# ---------------------------------------------------------------
# 7. Hermitian complex rank-deficient
# ---------------------------------------------------------------
Z = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
A = Z @ Z.conj().T  # rank 2 Hermitian PSD
report("Rank-2 Hermitian PSD (4x4)", A, hermitian=True)

# ---------------------------------------------------------------
# 8. Singular but symmetric saddle (zero block on diagonal)
#    [ 0  B ; B^T 0 ]  -- typical KKT structure
# ---------------------------------------------------------------
B = rng.standard_normal((3, 3))
A = np.block([[np.zeros((3, 3)), B], [B.T, np.zeros((3, 3))]])
report("Saddle-point matrix [[0,B],[B^T,0]] (6x6)", A)

# ---------------------------------------------------------------
# 9. Slightly perturbed rank-1 (numerically near singular)
# ---------------------------------------------------------------
v = np.array([1.0, 2.0, 3.0])
A = np.outer(v, v) + 1e-12 * np.eye(3)
report("Near-singular rank-1 + 1e-12 * I (3x3)", A)
