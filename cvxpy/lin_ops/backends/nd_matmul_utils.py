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

High-level helpers for backend unification:
- get_nd_matmul_dims: Compute batch_size, n, and whether constant has batch dims
- expand_lhs_for_nd_matmul: Expand constant lhs for ND matmul (handles both cases)
- expand_parametric_slices: Generator for expanding parametric lhs slices
"""

from typing import Iterator, Tuple

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


# =============================================================================
# High-level helpers for backend unification
# =============================================================================


def get_nd_matmul_dims(
    const_shape: Tuple[int, ...],
    var_shape: Tuple[int, ...],
) -> Tuple[int, int, bool]:
    """
    Compute dimensions for ND matmul C @ X.

    Parameters
    ----------
    const_shape : tuple
        Shape of the constant C
    var_shape : tuple
        Shape of the variable X

    Returns
    -------
    batch_size : int
        Product of batch dimensions from X (1 if X is 2D)
    n : int
        Last dimension of X (number of columns)
    const_has_batch : bool
        Whether C has batch dimensions (len > 2)
    """
    batch_size = int(np.prod(var_shape[:-2])) if len(var_shape) > 2 else 1
    n = var_shape[-1] if len(var_shape) >= 2 else 1
    const_has_batch = len(const_shape) > 2
    return batch_size, n, const_has_batch


def expand_lhs_for_nd_matmul(
    lhs_sparse: sp.sparray,
    const_data: np.ndarray,
    const_shape: Tuple[int, ...],
    var_shape: Tuple[int, ...],
) -> sp.sparray:
    """
    Expand constant lhs matrix for ND matmul.

    Handles both cases:
    - Batch-varying constant: uses interleaved matrix structure
    - 2D constant: uses I_n ⊗ C ⊗ I_batch Kronecker structure

    Parameters
    ----------
    lhs_sparse : sp.sparray
        The constant matrix C as sparse (used for 2D case)
    const_data : np.ndarray
        Raw constant data (used for batch-varying case)
    const_shape : tuple
        Shape of the constant
    var_shape : tuple
        Shape of the variable

    Returns
    -------
    sp.sparray
        The expanded matrix ready for multiplication with vec(X)
    """
    batch_size, n, const_has_batch = get_nd_matmul_dims(const_shape, var_shape)

    if const_has_batch:
        return build_interleaved_matrix(const_data, const_shape, var_shape)
    else:
        return apply_nd_kron_structure(lhs_sparse, batch_size, n)


def expand_parametric_slices(
    stacked_matrix: sp.sparray,
    param_size: int,
    batch_size: int,
    n: int,
) -> Iterator[sp.sparray]:
    """
    Generator yielding expanded slices for parametric ND matmul.

    For a stacked parameter matrix of shape (param_size * m, k), extracts each
    (m, k) slice and applies I_n ⊗ C ⊗ I_batch structure.

    Parameters
    ----------
    stacked_matrix : sp.sparray
        Stacked parameter matrix of shape (param_size * m, k)
    param_size : int
        Number of parameter slices
    batch_size : int
        Product of batch dimensions from the variable
    n : int
        Last dimension of the variable

    Yields
    ------
    sp.sparray
        Each expanded slice with shape (batch_size * m * n, batch_size * k * n)
    """
    m = stacked_matrix.shape[0] // param_size

    for slice_idx in range(param_size):
        slice_matrix = stacked_matrix[slice_idx * m:(slice_idx + 1) * m, :]
        yield apply_nd_kron_structure(slice_matrix, batch_size, n)
