"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from enum import Enum

import numpy as np
import scipy.sparse as sp


class TriangleKind(Enum):
    LOWER = "lower"
    UPPER = "upper"


def tri_to_full(tri_vec: np.ndarray, n: int, triangle: TriangleKind,
                sqrt2_scaling: bool) -> np.ndarray:
    """Expand vectorized triangle(s) to full symmetric matrix/matrices.

    Parameters
    ----------
    tri_vec : numpy.ndarray
        1-D array of triangular entries.  May contain a single triangle
        of length ``n*(n+1)//2``, or ``num`` contiguous triangles of total
        length ``num * n*(n+1)//2``.
    n : int
        The matrix side length.
    triangle : TriangleKind
        Which triangle ``tri_vec`` represents.
    sqrt2_scaling : bool
        If True, divide off-diagonal entries by sqrt(2).

    Returns
    -------
    numpy.ndarray
        If a single triangle was given, a shape ``(n, n)`` symmetric matrix.
        If multiple triangles were given, shape ``(num, n, n)``.

    Notes
    -----
    SCS and Clarabel track triangular indices in a transposed way relative to
    NumPy's convention, so `LOWER` uses ``np.triu_indices`` and `UPPER` uses
    ``np.tril_indices``. This looks wrong but is correct.
    """
    tri_dim = n * (n + 1) // 2
    num = tri_vec.size // tri_dim
    full = np.zeros((num, n, n))
    tri_vecs = tri_vec.reshape(num, tri_dim)
    tri_idx = np.triu_indices(n) if triangle == TriangleKind.LOWER else np.tril_indices(n)
    full[:, *tri_idx] = tri_vecs
    full += np.swapaxes(full, -2, -1)
    diag_idx = np.diag_indices(n)
    full[:, *diag_idx] /= 2
    if sqrt2_scaling:
        lo = np.tril_indices(n, k=-1)
        hi = np.triu_indices(n, k=1)
        full[:, *lo] /= np.sqrt(2)
        full[:, *hi] /= np.sqrt(2)
    if num == 1:
        return full[0]
    return full


def psd_format_mat(constr, triangle: TriangleKind,
                   sqrt2_scaling: bool) -> sp.csc_array:
    """Return a sparse matrix that extracts the svec of each PSD matrix.

    Given a PSD constraint whose argument has shape ``(*batch, n, n)``,
    build a sparse matrix ``M`` such that ``M @ vec(X, order='F')``
    produces the scaled vectorized triangle of each matrix in the batch.

    For batched constraints the F-order vec interleaves batch elements,
    so a permutation is included to de-interleave before the per-block
    triangle extraction.

    Parameters
    ----------
    constr : PSD
        The PSD constraint.
    triangle : TriangleKind
        Which triangle the solver expects.
    sqrt2_scaling : bool
        Whether off-diagonal entries are scaled by sqrt(2).
    """
    n = constr.args[0].shape[-1]
    entries = n * (n + 1) // 2

    # --- Build the single-block format matrix (entries × n²) ---
    row_arr = np.arange(entries)
    if triangle == TriangleKind.LOWER:
        tri_idx = np.tril_indices(n)
    else:
        tri_idx = np.triu_indices(n)
    col_arr = np.sort(np.ravel_multi_index(tri_idx, (n, n), order='F'))

    val_arr = np.zeros((n, n))
    val_arr[tri_idx] = np.sqrt(2) if sqrt2_scaling else 1.0
    np.fill_diagonal(val_arr, 1.0)
    val_arr = val_arr.ravel(order='F')
    val_arr = val_arr[np.nonzero(val_arr)]

    scaled_tri = sp.csc_array((val_arr, (row_arr, col_arr)), (entries, n * n))

    idx = np.arange(n * n)
    K = idx.reshape((n, n))
    row_symm = np.append(idx, K.ravel(order='F'))
    col_symm = np.append(idx, K.T.ravel(order='F'))
    val_symm = 0.5 * np.ones(2 * n * n)
    symm_matrix = sp.csc_array((val_symm, (row_symm, col_symm)))

    single_block = scaled_tri @ symm_matrix

    # --- Handle batched constraints ---
    num = constr.num_cones()
    if num == 1:
        return single_block

    block_format = sp.block_diag([single_block] * num, format='csc')

    # F-order vec of (*batch, n, n) interleaves batch elements.
    # Permute from interleaved to contiguous per-batch blocks.
    nn = n * n
    total = num * nn
    perm = np.arange(total).reshape(nn, num).T.ravel()
    perm_mat = sp.csc_array(
        (np.ones(total), (perm, np.arange(total))),
        shape=(total, total))

    return block_format @ perm_mat.T
