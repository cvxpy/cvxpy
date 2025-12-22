"""
Utility functions for ND matrix multiplication operations in canonicalization backends.

These functions implement the core matrix transformations needed for ND matmul:
- build_interleaved_matrix: For batch-varying constant case (C has batch dims)
- apply_nd_kron_structure: For I_n ⊗ C ⊗ I_batch structure (C is 2D, X has batch dims)

The mathematical background:
- For vectorized matmul C @ X with Fortran-order vectorization, we use Kronecker products
- 2D case: vec(C @ X) = (I_n ⊗ C) @ vec(X)
- ND case with 2D constant: vec(C @ X) = (I_n ⊗ C ⊗ I_batch) @ vec(X)
- ND case with batch-varying constant: uses interleaved matrix structure
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp


def build_interleaved_matrix(
    const_data: np.ndarray,
    const_shape: Tuple[int, ...],
    var_shape: Tuple[int, ...],
) -> sp.csr_array:
    """
    Build the interleaved matrix for batch-varying constant case.

    For C (..., m, k) @ X (..., k, n), builds I_n ⊗ M_interleaved where:
    M_interleaved[b + B*i, b + B*r] = C[b, i, r]

    This captures the Fortran-order vectorization where batch indices are interleaved:
    - result[b, i, c] is at index b + B*i + B*m*c
    - X[b, r, c] is at index b + B*r + B*k*c

    Parameters
    ----------
    const_data : np.ndarray
        Raw constant data (will be reshaped to (B, m, k) in Fortran order)
    const_shape : tuple
        Shape of the constant (..., m, k)
    var_shape : tuple
        Shape of the variable (..., k, n)

    Returns
    -------
    sp.csr_array
        The stacked matrix I_n ⊗ M_interleaved
    """
    B = int(np.prod(const_shape[:-2]))
    m_dim = const_shape[-2]
    k_dim = const_shape[-1]
    n = var_shape[-1]

    # Reshape to (B, m, k) in Fortran order
    const_flat = np.reshape(const_data, (B, m_dim, k_dim), order="F")

    # Build interleaved matrix using vectorized operations
    # Entry (b + B*i, b + B*r) = C[b, i, r]
    b_indices = np.arange(B)
    i_indices = np.arange(m_dim)
    r_indices = np.arange(k_dim)

    # Create meshgrid for all combinations (B, m, k)
    bb, ii, rr = np.meshgrid(b_indices, i_indices, r_indices, indexing="ij")

    rows = (bb + B * ii).ravel()
    cols = (bb + B * rr).ravel()
    data = const_flat.ravel()

    M_interleaved = sp.csr_array((data, (rows, cols)), shape=(B * m_dim, B * k_dim))

    # Apply I_n ⊗ M_interleaved
    if n > 1:
        return sp.kron(sp.eye_array(n, format="csr"), M_interleaved)
    return M_interleaved


def apply_nd_kron_structure(
    lhs: sp.sparray,
    batch_size: int,
    n: int,
) -> sp.sparray:
    """
    Apply ND Kronecker structure I_n ⊗ C ⊗ I_batch.

    For ND matmul C @ X where C is 2D (m, k) and X has shape (..., k, n):
    vec(C @ X) = (I_n ⊗ C ⊗ I_batch) @ vec(X)
    where batch = prod(...).

    Parameters
    ----------
    lhs : sp.sparray
        The (m, k) matrix C (sparse)
    batch_size : int
        Product of batch dimensions from X
    n : int
        Last dimension of X

    Returns
    -------
    sp.sparray
        The expanded matrix I_n ⊗ C ⊗ I_batch
    """
    if batch_size > 1:
        inner = sp.kron(lhs, sp.eye_array(batch_size, format="csr"))
    else:
        inner = lhs

    if n > 1:
        return sp.kron(sp.eye_array(n, format="csr"), inner)
    return inner
