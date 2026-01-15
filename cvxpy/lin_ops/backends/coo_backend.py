"""
Copyright 2025, the CVXPY authors.

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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops.backends.base import (
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
    get_nd_matmul_dims,
    get_nd_rmul_dims,
    is_batch_varying,
)

# Module-level empty array constants to avoid repeated allocations
_EMPTY_FLOAT = np.array([], dtype=np.float64)
_EMPTY_INT = np.array([], dtype=np.int64)


def _empty_float() -> np.ndarray:
    """Return a copy of an empty float64 array."""
    return _EMPTY_FLOAT.copy()


def _empty_int() -> np.ndarray:
    """Return a copy of an empty int64 array."""
    return _EMPTY_INT.copy()


def compute_indptr(indices: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CSR/CSC-style indptr directly from COO indices.

    This avoids creating scipy sparse matrices just to get the indptr array.
    Returns (indptr, sort_perm) where:
    - sort_perm: permutation that sorts data by the given indices
    - indptr: array of length size+1 where indptr[i]:indptr[i+1] gives
              the range for index i in the sorted data

    Example:
        indices = [2, 0, 2, 1]  # row indices
        size = 3
        Returns:
            sort_perm = [1, 3, 0, 2]  # sorted order
            indptr = [0, 1, 2, 4]     # row 0: [0:1], row 1: [1:2], row 2: [2:4]
    """
    if len(indices) == 0:
        return np.zeros(size + 1, dtype=np.int64), np.array([], dtype=np.int64)

    sort_perm = np.argsort(indices)
    sorted_indices = indices[sort_perm]

    # Count entries per index using bincount
    counts = np.bincount(sorted_indices, minlength=size)

    # Cumsum to get indptr
    indptr = np.zeros(size + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    return indptr, sort_perm


@dataclass
class CooTensor:
    """
    Compact representation of a parameter-indexed sparse tensor.

    Represents the 3D tensor T[param_idx, row, col] in COO format.

    For a tensor with shape (param_size, m, n):
    - data[i] is the value at position (param_idx[i], row[i], col[i])
    - All arrays have the same length (nnz)

    This is equivalent to TensorRepresentation but without the shape constraint.
    """
    data: np.ndarray       # Values (nnz,)
    row: np.ndarray        # Row indices within each slice (nnz,) range [0, m)
    col: np.ndarray        # Column indices (nnz,) range [0, n)
    param_idx: np.ndarray  # Parameter slice index (nnz,) range [0, param_size)
    m: int                 # Number of rows per slice
    n: int                 # Number of columns
    param_size: int        # Number of parameter slices

    @property
    def nnz(self) -> int:
        return len(self.data)

    @property
    def stacked_shape(self) -> tuple[int, int]:
        """Shape if this were converted to stacked format."""
        return (self.param_size * self.m, self.n)

    def to_stacked_sparse(self) -> sp.csr_array:
        """Convert to stacked sparse matrix (for compatibility)."""
        stacked_rows = self.param_idx * self.m + self.row
        return sp.csr_array(
            (self.data, (stacked_rows, self.col)),
            shape=self.stacked_shape
        )

    # Sparse matrix compatibility methods
    def tocoo(self) -> sp.coo_array:
        """Convert to COO format (stacked) for compatibility."""
        return self.to_stacked_sparse().tocoo()

    def tocsr(self) -> sp.csr_array:
        """Convert to CSR format (stacked) for compatibility."""
        return self.to_stacked_sparse().tocsr()

    def tocsc(self) -> sp.csc_array:
        """Convert to CSC format (stacked) for compatibility."""
        return self.to_stacked_sparse().tocsc()

    def toarray(self) -> np.ndarray:
        """Convert to dense array (stacked) for compatibility."""
        return self.to_stacked_sparse().toarray()

    @property
    def shape(self) -> tuple[int, int]:
        """Return stacked shape for compatibility."""
        return self.stacked_shape

    @classmethod
    def empty(cls, m: int, n: int, param_size: int) -> CooTensor:
        """Create an empty CooTensor with given dimensions."""
        return cls(
            data=_empty_float(),
            row=_empty_int(),
            col=_empty_int(),
            param_idx=_empty_int(),
            m=m,
            n=n,
            param_size=param_size
        )

    @classmethod
    def from_stacked_sparse(cls, matrix: sp.spmatrix, param_size: int) -> CooTensor:
        """Convert from stacked sparse matrix."""
        coo = matrix.tocoo()
        m = matrix.shape[0] // param_size
        n = matrix.shape[1]

        param_idx, row = np.divmod(coo.row, m)

        return cls(
            data=coo.data.copy(),
            row=row,
            col=coo.col.copy(),
            param_idx=param_idx,
            m=m,
            n=n,
            param_size=param_size
        )

    def select_rows(self, rows: np.ndarray) -> CooTensor:
        """
        Select and reorder rows from each parameter slice.

        Semantics match SciPy: new_tensor[i, :] = old_tensor[rows[i], :]
        This means `rows` contains the SOURCE row index for each DESTINATION row.

        For example, rows=[2, 0, 1] means:
          - new row 0 <- old row 2
          - new row 1 <- old row 0
          - new row 2 <- old row 1

        Duplicate Handling (Broadcasting)
        ----------------------------------
        When `rows` contains duplicate values, the same source row is copied to
        multiple destination rows. This occurs during broadcasting operations.

        Example: A parameter P of shape (2, 3) broadcast to (4, 2, 3):
          - Original rows: [0, 1, 2, 3, 4, 5] (6 elements)
          - After broadcast: [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, ...] (24 elements)
          - Each original row appears 4 times (once per batch)

        For parametric tensors, this means the same param_idx values get replicated
        to multiple output positions - which is correct because the same parameter
        value is used in multiple places after broadcasting.
        (See broadcast_to() and reshape_parametric_constant() for deduplication.)

        Parameters
        ----------
        rows : np.ndarray
            Array of source row indices. Length determines the number of output rows.

        Returns
        -------
        CooTensor
            New tensor with selected/reordered rows.
        """
        new_m = len(rows)

        if self.nnz == 0:
            return CooTensor.empty(new_m, self.n, self.param_size)

        # Check for duplicates - avoid sorting if already sorted
        diffs = np.diff(rows)
        has_duplicates = (np.any(diffs == 0) if np.all(diffs >= 0)
                          else np.any(np.diff(np.sort(rows)) == 0))
        # Fast path: no duplicate rows (used by transpose, simple indexing)
        if not has_duplicates:
            return self._select_rows_no_duplicates(rows, new_m)
        # General path: handles duplicate rows (used by broadcast_to, promote)
        else:
            return self._select_rows_with_duplicates(rows, new_m)

    def _select_rows_no_duplicates(self, rows: np.ndarray, new_m: int) -> CooTensor:
        """
        Fast path for select_rows when there are no duplicate rows.

        Uses a reverse mapping approach: for each old row, compute which new row
        it maps to (or -1 if not selected). This is O(nnz) and avoids sorting.

        Example: Select rows [2, 0] from a tensor with 3 rows
        -----------------------------------------------------
        rows = [2, 0] means: new_row_0 <- old_row_2, new_row_1 <- old_row_0

        Build reverse map (old_row -> new_row):
            row_map = [-1, -1, -1]  (initialize: nothing selected)
            row_map[2] = 0  (old row 2 -> new row 0)
            row_map[0] = 1  (old row 0 -> new row 1)
            row_map = [1, -1, 0]

        Apply to tensor entries:
            If self.row = [0, 1, 2, 0, 2]
            new_rows = row_map[self.row] = [1, -1, 0, 1, 0]
            mask = [True, False, True, True, True]  (keep where >= 0)

        Result: entries from old rows 0,2 are kept and renumbered.
        """
        # Build reverse mapping: old_row -> new_row (or -1 if not selected)
        row_map = np.full(self.m, -1, dtype=np.int64)
        row_map[rows] = np.arange(new_m, dtype=np.int64)

        # Map tensor rows to new positions
        new_rows = row_map[self.row]

        # Keep only entries whose row is selected
        mask = new_rows >= 0

        return CooTensor(
            data=self.data[mask],
            row=new_rows[mask],
            col=self.col[mask],
            param_idx=self.param_idx[mask],
            m=new_m,
            n=self.n,
            param_size=self.param_size
        )

    def _select_rows_with_duplicates(self, rows: np.ndarray, new_m: int) -> CooTensor:
        """
        General path for select_rows that handles duplicate rows.

        This method is used when broadcasting causes the same source row to be
        copied to multiple destination rows. Uses binary search to find all
        destination positions for each tensor entry, then replicates entries
        accordingly. O(nnz log n) complexity.

        When to Use This Path
        ---------------------
        - broadcast_to: Parameter (2,3) -> (4,2,3) creates rows with duplicates
        - promote: Scalar broadcast to vector creates all-same rows

        Example: Broadcast parameter from shape (2,) to (3, 2)
        ------------------------------------------------------
        Original parameter tensor (2 elements, param_size=2):
            data = [1, 1]
            row = [0, 1]           # param element positions
            param_idx = [0, 1]    # each element is unique param slice

        broadcast_to creates rows = [0, 1, 0, 1, 0, 1]
        (3 copies of the 2-element vector, flattened column-major)

        Algorithm steps:
        1. Sort rows: row_sort_perm = [0, 2, 4, 1, 3, 5], rows_sorted = [0,0,0,1,1,1]
        2. For each unique tensor row, count occurrences in rows:
           - Tensor row 0 appears at positions 0,2,4 in rows -> count=3
           - Tensor row 1 appears at positions 1,3,5 in rows -> count=3
        3. Replicate each tensor entry by its count:
           - Entry (row=0, param_idx=0) replicated 3 times
           - Entry (row=1, param_idx=1) replicated 3 times
        4. Compute output row indices from row_sort_perm

        Result:
            data = [1, 1, 1, 1, 1, 1]  (6 entries)
            row = [0, 2, 4, 1, 3, 5]   (destination positions)
            param_idx = [0, 0, 0, 1, 1, 1]  <- DUPLICATED! Same param used 3x

        The duplicated param_idx values are correct: after broadcasting, the same
        parameter value contributes to multiple output positions. At solve time,
        when we multiply each param slice by its value, these duplicated entries
        will all receive the same parameter value.
        """
        # Sort rows to enable binary search for range queries
        row_sort_perm = np.argsort(rows)
        rows_sorted = rows[row_sort_perm]

        # Find unique tensor rows and count how many times each appears in `rows`
        # This tells us how many output entries each tensor entry will generate
        unique_tensor_rows, inverse_idx = np.unique(self.row, return_inverse=True)
        left = np.searchsorted(rows_sorted, unique_tensor_rows, side='left')
        right = np.searchsorted(rows_sorted, unique_tensor_rows, side='right')
        counts = right - left  # Number of occurrences of each unique row in `rows`

        # Compute per-entry replication counts
        entry_counts = counts[inverse_idx]
        total_nnz = entry_counts.sum()
        if total_nnz == 0:
            return CooTensor.empty(new_m, self.n, self.param_size)

        # Replicate data, col, param_idx arrays according to counts
        # This is where param_idx gets duplicated for broadcast parameters
        out_data = np.repeat(self.data, entry_counts)
        out_col = np.repeat(self.col, entry_counts)
        out_param = np.repeat(self.param_idx, entry_counts)

        # Build output row indices by gathering from row_sort_perm
        # For each tensor entry, we need to output to all positions where its
        # row value appears in `rows`. These positions are row_sort_perm[left:right].
        entry_lefts = np.repeat(left[inverse_idx], entry_counts)
        offsets = np.concatenate([[0], np.cumsum(entry_counts)[:-1]])
        positions = np.arange(total_nnz, dtype=np.int64) - np.repeat(offsets, entry_counts)
        out_row = row_sort_perm[entry_lefts + positions]

        return CooTensor(
            data=out_data,
            row=out_row,
            col=out_col,
            param_idx=out_param,
            m=new_m,
            n=self.n,
            param_size=self.param_size
        )

    def scale(self, factor: float) -> CooTensor:
        """Scale all values by a constant."""
        return CooTensor(
            data=self.data * factor,
            row=self.row,
            col=self.col,
            param_idx=self.param_idx,
            m=self.m,
            n=self.n,
            param_size=self.param_size
        )

    def negate(self) -> CooTensor:
        """Negate all values."""
        return self.scale(-1.0)

    def __neg__(self) -> CooTensor:
        """Support unary negation (-tensor)."""
        return self.negate()

    def _transpose_helper(self) -> CooTensor:
        """2D matrix transpose (swap rows and cols).

        Internal helper for matrix operations like rmul. Not for ND linop transpose.
        """
        return CooTensor(
            data=self.data,
            row=self.col,
            col=self.row,
            param_idx=self.param_idx,
            m=self.n,
            n=self.m,
            param_size=self.param_size
        )

    def __add__(self, other: CooTensor) -> CooTensor:
        """Add two CooTensors (concatenate entries)."""
        if self.m != other.m:
            raise ValueError(f"Row count mismatch: {self.m} vs {other.m}")
        if self.n != other.n:
            raise ValueError(f"Column count mismatch: {self.n} vs {other.n}")
        if self.param_size != other.param_size:
            raise ValueError(f"Param size mismatch: {self.param_size} vs {other.param_size}")

        return CooTensor(
            data=np.concatenate([self.data, other.data]),
            row=np.concatenate([self.row, other.row]),
            col=np.concatenate([self.col, other.col]),
            param_idx=np.concatenate([self.param_idx, other.param_idx]),
            m=self.m,
            n=self.n,
            param_size=self.param_size
        )


def _kron_eye_r(tensor: CooTensor, reps: int) -> CooTensor:
    """
    Apply Kronecker product kron(I_reps, A) to a CooTensor.

    For a tensor with shape (m, k), this produces a tensor with shape (m*reps, k*reps)
    where the original matrix is replicated along the block diagonal.

    Each entry at (param_idx, row, col) becomes `reps` entries at
    (param_idx, row + r*m, col + r*k) for r in 0..reps-1.
    """
    if reps == 1:
        return tensor

    m, k = tensor.m, tensor.n
    nnz = tensor.nnz

    # Each original entry expands to reps entries
    new_data = np.tile(tensor.data, reps)
    new_param_idx = np.tile(tensor.param_idx, reps)

    # Row and col offsets for each rep
    row_offsets = np.repeat(np.arange(reps) * m, nnz)
    col_offsets = np.repeat(np.arange(reps) * k, nnz)

    new_row = np.tile(tensor.row, reps) + row_offsets
    new_col = np.tile(tensor.col, reps) + col_offsets

    return CooTensor(
        data=new_data,
        row=new_row,
        col=new_col,
        param_idx=new_param_idx,
        m=m * reps,
        n=k * reps,
        param_size=tensor.param_size
    )


def _kron_eye_l(tensor: CooTensor, reps: int) -> CooTensor:
    """
    Apply Kronecker product kron(A, I_reps) to a CooTensor.

    For a tensor with shape (m, k), this produces a tensor with shape (m*reps, k*reps)
    where each entry at (param_idx, row, col) becomes `reps` entries at
    (param_idx, row*reps + r, col*reps + r) for r in 0..reps-1.

    This creates a block diagonal structure where each scalar in A becomes a
    diagonal block of size reps x reps.
    """
    if reps == 1:
        return tensor

    m, k = tensor.m, tensor.n
    nnz = tensor.nnz

    # Each original entry expands to reps entries
    new_data = np.tile(tensor.data, reps)
    new_param_idx = np.tile(tensor.param_idx, reps)

    # new_row = old_row * reps + offset
    # new_col = old_col * reps + offset
    offsets = np.repeat(np.arange(reps), nnz)
    new_row = np.tile(tensor.row * reps, reps) + offsets
    new_col = np.tile(tensor.col * reps, reps) + offsets

    return CooTensor(
        data=new_data,
        row=new_row,
        col=new_col,
        param_idx=new_param_idx,
        m=m * reps,
        n=k * reps,
        param_size=tensor.param_size
    )


def _kron_nd_structure_mul(tensor: CooTensor, batch_size: int, n: int) -> CooTensor:
    """
    Build the Kronecker structure for ND mul (C @ X): I_n ⊗ C ⊗ I_B.

    This is the key operation for computing C @ X where C is a 2D constant (m, k)
    and X is an ND variable with batch dimensions (..., k, n).

    Why This Structure?
    -------------------
    For batched matmul C(m,k) @ X(B,k,n) = Y(B,m,n), each output element is:

        Y[b, i, j] = Σ_r C[i, r] * X[b, r, j]

    When we vectorize in column-major (Fortran) order:
        - vec(X) has index: b + B*r + B*k*j  (b fastest, then r, then j)
        - vec(Y) has index: b + B*i + B*m*j  (b fastest, then i, then j)

    The matrix A where vec(Y) = A @ vec(X) must satisfy:
        A[b + B*i + B*m*j, b' + B*r + B*k*j'] = C[i, r]  if b==b' and j==j'
                                              = 0        otherwise

    This sparsity pattern is exactly I_n ⊗ C ⊗ I_B:
        - I_B: same batch index (b == b') - diagonal in batch dimension
        - C:   the actual matmul coefficients C[i, r]
        - I_n: same output column (j == j') - diagonal in output column dimension

    Concrete Example
    ----------------
    C = [[1, 2, 3],    shape (2, 3)
         [4, 5, 6]]

    X has shape (B=2, k=3, n=2):
        X[0,:,:] = [[x00, x01],    X[1,:,:] = [[x10, x11],
                    [x02, x03],                 [x12, x13],
                    [x04, x05]]                 [x14, x15]]

    Result Y = C @ X has shape (2, 2, 2):
        Y[b, i, j] = C[i, 0]*X[b,0,j] + C[i, 1]*X[b,1,j] + C[i, 2]*X[b,2,j]

    Vectorized (F-order): vec(X) = [x00,x10, x02,x12, x04,x14, x01,x11, ...]
                                    └─b=0,1─┘ └─b=0,1─┘ └─b=0,1─┘ └─ j=1 ─...
                                      r=0       r=1       r=2

    The matrix A = I_2 ⊗ C ⊗ I_2 has shape (2*2*2, 2*3*2) = (8, 12):

        Block structure (showing j=0 block, j=1 block is identical):
        ┌─────────────────────────────────────┐
        │ C⊗I_2   0       0      │    0       │  ← j=0 output
        │   0    C⊗I_2    0      │    0       │
        │   0     0     C⊗I_2    │    0       │
        ├─────────────────────────────────────┤
        │   0     0       0      │  C⊗I_2 ...│  ← j=1 output
        └─────────────────────────────────────┘
              j=0 input              j=1 input

        Where C⊗I_2 for the (2,3) matrix C is (4, 6):
        ┌───────────────────┐
        │ 1 0 │ 2 0 │ 3 0  │  ← i=0, b=0,1
        │ 0 1 │ 0 2 │ 0 3  │
        │─────┼─────┼──────│
        │ 4 0 │ 5 0 │ 6 0  │  ← i=1, b=0,1
        │ 0 4 │ 0 5 │ 0 6  │
        └───────────────────┘
          r=0   r=1   r=2

    Implementation
    --------------
    Equivalent to: _kron_eye_r(_kron_eye_l(tensor, batch_size), n)
    but computed in a single pass for efficiency.

    Each entry at (row=i, col=j) in C expands to batch_size * n entries at:
        (row = B*(i + r*m) + b, col = B*(j + r*k) + b)
    for b in [0, batch_size) and r in [0, n).

    Parameters
    ----------
    tensor : CooTensor
        The input tensor C with shape (param_size, m, k)
    batch_size : int
        The batch dimension B (= prod of batch dims from variable)
    n : int
        The last dimension of the variable (output columns)

    Returns
    -------
    CooTensor
        Expanded tensor with shape (param_size, m*B*n, k*B*n)
    """
    if batch_size == 1 and n == 1:
        return tensor
    if batch_size == 1:
        return _kron_eye_r(tensor, n)
    if n == 1:
        return _kron_eye_l(tensor, batch_size)

    m, k = tensor.m, tensor.n
    nnz = tensor.nnz
    expansion = batch_size * n

    # Pre-compute all (b, r) combinations
    # b_flat iterates slowest: [0,0,...,0, 1,1,...,1, ...]
    # r_flat iterates fastest: [0,1,...,n-1, 0,1,...,n-1, ...]
    b_vals, r_vals = np.meshgrid(np.arange(batch_size), np.arange(n), indexing='ij')
    b_flat = b_vals.ravel()
    r_flat = r_vals.ravel()

    # Expand arrays: each entry replicates expansion times
    new_data = np.tile(tensor.data, expansion)
    new_param_idx = np.tile(tensor.param_idx, expansion)

    # new_row = B*(i + r*m) + b
    # new_col = B*(j + r*k) + b
    base_row = np.tile(tensor.row, expansion)
    base_col = np.tile(tensor.col, expansion)

    b_offsets = np.repeat(b_flat, nnz)
    r_offsets = np.repeat(r_flat, nnz)

    new_row = batch_size * (base_row + r_offsets * m) + b_offsets
    new_col = batch_size * (base_col + r_offsets * k) + b_offsets

    return CooTensor(
        data=new_data,
        row=new_row,
        col=new_col,
        param_idx=new_param_idx,
        m=m * batch_size * n,
        n=k * batch_size * n,
        param_size=tensor.param_size
    )


def _build_interleaved_mul(
    const_data: np.ndarray,
    const_shape: tuple,
    var_shape: tuple,
) -> CooTensor:
    """
    Build interleaved matrix for batch-varying mul (C @ X).

    This handles the case where each batch element uses a DIFFERENT constant
    matrix: C(B,m,k) @ X(B,k,n) = Y(B,m,n) where C[b] differs for each b.

    Why Not Use Kronecker?
    ----------------------
    The Kronecker structure I_n ⊗ C ⊗ I_B (from _kron_nd_structure_mul) assumes
    the SAME matrix C is applied to all batches. It creates a block-diagonal
    structure where C appears repeatedly.

    But with batch-varying constants, each batch needs its OWN coefficients:
        Y[b, i, j] = Σ_r C[b, i, r] * X[b, r, j]
                         ↑
                      Different C for each b!

    The Interleaved Structure
    -------------------------
    We need matrix A where vec(Y) = A @ vec(X) with:
        A[b + B*i + B*m*j, b' + B*r + B*k*j'] = C[b, i, r]  if b==b' and j==j'

    The key insight is the indexing pattern: instead of block-diagonal,
    we INTERLEAVE the batch dimension with the matrix dimensions:
        M_interleaved[b + B*i, b + B*r] = C[b, i, r]

    This places each batch's coefficients at positions that only interact
    with that batch's input/output elements.

    Concrete Example
    ----------------
    C has shape (B=2, m=2, k=2):
        C[0] = [[a, b],      C[1] = [[e, f],
                [c, d]]              [g, h]]

    X has shape (B=2, k=2, n=1), so output Y has shape (B=2, m=2, n=1).

    Vectorization (F-order, n=1 so j dimension is trivial):
        vec(X) = [X[0,0,0], X[1,0,0], X[0,1,0], X[1,1,0]]
               = [x0,       x1,       x2,       x3      ]
                  └─b=0,1──┘ └──b=0,1──┘
                    r=0         r=1

        vec(Y) = [Y[0,0,0], Y[1,0,0], Y[0,1,0], Y[1,1,0]]
               = [y0,       y1,       y2,       y3      ]
                  └─b=0,1──┘ └──b=0,1──┘
                    i=0         i=1

    The computation:
        y0 = Y[0,0,0] = C[0,0,0]*X[0,0,0] + C[0,0,1]*X[0,1,0] = a*x0 + b*x2
        y1 = Y[1,0,0] = C[1,0,0]*X[1,0,0] + C[1,0,1]*X[1,1,0] = e*x1 + f*x3
        y2 = Y[0,1,0] = C[0,1,0]*X[0,0,0] + C[0,1,1]*X[0,1,0] = c*x0 + d*x2
        y3 = Y[1,1,0] = C[1,1,0]*X[1,0,0] + C[1,1,1]*X[1,1,0] = g*x1 + h*x3

    Interleaved matrix M (4×4):
                x0  x1  x2  x3
            ┌───────────────────┐
        y0  │  a   0   b   0   │   C[0,0,0]=a at (0,0), C[0,0,1]=b at (0,2)
        y1  │  0   e   0   f   │   C[1,0,0]=e at (1,1), C[1,0,1]=f at (1,3)
        y2  │  c   0   d   0   │   C[0,1,0]=c at (2,0), C[0,1,1]=d at (2,2)
        y3  │  0   g   0   h   │   C[1,1,0]=g at (3,1), C[1,1,1]=h at (3,3)
            └───────────────────┘

    Pattern: M[b + B*i, b + B*r] = C[b, i, r]
        - batch 0 elements (a,b,c,d) are at even rows/cols
        - batch 1 elements (e,f,g,h) are at odd rows/cols
        - No cross-batch interactions (zeros where b ≠ b')

    Contrast with Kronecker (if C were NOT batch-varying):
        Kronecker would put C in blocks: [[C,0],[0,C]]
        Interleaved disperses C's elements: rows/cols alternate by batch

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
    CooTensor
        The interleaved matrix with I_n applied (for n > 1)
    """
    B = int(np.prod(const_shape[:-2]))
    m = const_shape[-2]
    k = const_shape[-1]
    n = var_shape[-1]

    # Reshape to (B, m, k) in Fortran order
    const_flat = np.reshape(const_data, (B, m, k), order="F")

    # Build interleaved indices for all (b, i, r) combinations
    b_idx = np.arange(B)
    i_idx = np.arange(m)
    r_idx = np.arange(k)

    # bb, ii, rr are grids of indices corresponding to b, i, r in the formula below
    bb, ii, rr = np.meshgrid(b_idx, i_idx, r_idx, indexing="ij")
    bb, ii, rr = bb.ravel(), ii.ravel(), rr.ravel()

    # M_interleaved[b + B*i, b + B*r] = C[b, i, r]
    base_row = bb + B * ii
    base_col = bb + B * rr
    data = const_flat.ravel()

    # Apply I_n ⊗ M_interleaved: replicate for each c in 0..n-1
    if n > 1:
        c_offsets = np.arange(n)
        row_offsets = np.repeat(c_offsets * B * m, len(data))
        col_offsets = np.repeat(c_offsets * B * k, len(data))

        new_data = np.tile(data, n)
        new_row = np.tile(base_row, n) + row_offsets
        new_col = np.tile(base_col, n) + col_offsets
    else:
        new_data = data
        new_row = base_row
        new_col = base_col

    return CooTensor(
        data=new_data.astype(np.float64),
        row=new_row.astype(np.int64),
        col=new_col.astype(np.int64),
        param_idx=np.zeros(len(new_data), dtype=np.int64),
        m=B * m * n,
        n=B * k * n,
        param_size=1
    )


def _kron_nd_structure_rmul(tensor: CooTensor, batch_size: int, m: int) -> CooTensor:
    """
    Build the Kronecker structure for ND rmul: C.T tensor I_{B*m}.

    This is the key operation for computing X @ C where C is a 2D constant (k, n)
    and X is an ND variable with batch dimensions (..., m, k).

    Why This Structure?
    -------------------
    For batched rmul X(B,m,k) @ C(k,n) = Y(B,m,n), each output element is:

        Y[b, i, j] = sum_r X[b, i, r] * C[r, j]

    When we vectorize in column-major (Fortran) order:
        - vec(X) has index: b + B*i + B*m*r  (b fastest, then i, then r)
        - vec(Y) has index: b + B*i + B*m*j  (b fastest, then i, then j)

    The matrix A where vec(Y) = A @ vec(X) must satisfy:
        A[b + B*i + B*m*j, b' + B*i' + B*m*r] = C[r, j]  if b==b' and i==i'
                                              = 0        otherwise

    This sparsity pattern is exactly C.T tensor I_{B*m}:
        - C.T provides the matmul coefficients C.T[j, r] = C[r, j]
        - I_{B*m}: same batch index AND same row index (b==b' and i==i')

    Concrete Example
    ----------------
    C = [[1, 2],    shape (k=2, n=2) so C.T = [[1, 3],
         [3, 4]]                               [2, 4]]

    X has shape (B=2, m=2, k=2):
        X[0,:,:] = [[x00, x01],    X[1,:,:] = [[x10, x11],
                    [x02, x03]]                [x12, x13]]

    Result Y = X @ C has shape (B=2, m=2, n=2):
        Y[b, i, j] = X[b,i,0]*C[0,j] + X[b,i,1]*C[1,j]

    Vectorized (F-order): vec(X) = [x00,x10, x02,x12, x01,x11, x03,x13]
                                    └─b=0,1─┘ └─b=0,1─┘ └─b=0,1─┘ └─b=0,1─┘
                                      i=0,r=0   i=1,r=0   i=0,r=1   i=1,r=1

    The matrix A = C.T ⊗ I_{B*m} = C.T ⊗ I_4 has shape (B*m*n, B*m*k) = (8, 8):

        Block structure (C.T ⊗ I_4):
        ┌─────────────────────────────┐
        │ C.T[0,0]*I_4 │ C.T[0,1]*I_4 │   = │ 1*I_4 │ 2*I_4 │
        │──────────────┼──────────────│     │───────┼───────│
        │ C.T[1,0]*I_4 │ C.T[1,1]*I_4 │     │ 3*I_4 │ 4*I_4 │
        └─────────────────────────────┘

        Full 8×8 matrix:
                 r=0,b=0,1,i=0,1  r=1,b=0,1,i=0,1
                 x00 x10 x02 x12  x01 x11 x03 x13
             ┌────────────────────────────────────┐
        j=0  │  1   0   0   0  │  2   0   0   0   │
        j=0  │  0   1   0   0  │  0   2   0   0   │
        j=0  │  0   0   1   0  │  0   0   2   0   │
        j=0  │  0   0   0   1  │  0   0   0   2   │
             │─────────────────┼──────────────────│
        j=1  │  3   0   0   0  │  4   0   0   0   │
        j=1  │  0   3   0   0  │  0   4   0   0   │
        j=1  │  0   0   3   0  │  0   0   4   0   │
        j=1  │  0   0   0   3  │  0   0   0   4   │
             └────────────────────────────────────┘

    Each identity block I_4 ensures the same (b,i) pair in X maps to
    the same (b,i) pair in Y, while C.T provides the matmul coefficients.

    Parameters
    ----------
    tensor : CooTensor
        The input tensor C with shape (param_size, k, n)
    batch_size : int
        The batch dimension B (= prod of batch dims from variable)
    m : int
        The second-to-last dimension of the variable (X's rows)

    Returns
    -------
    CooTensor
        Expanded tensor with shape (param_size, n*B*m, k*B*m)
    """
    reps = batch_size * m
    if reps == 1:
        return tensor._transpose_helper()

    # First transpose C to get C.T, then apply kron(C.T, I_{B*m})
    transposed = tensor._transpose_helper()
    return _kron_eye_l(transposed, reps)


def _build_interleaved_rmul(
    const_data: np.ndarray,
    const_shape: tuple,
    var_shape: tuple,
) -> CooTensor:
    """
    Build interleaved matrix for batch-varying rmul case.

    This handles the case where each batch element uses a DIFFERENT constant
    matrix: X(B,m,k) @ C(B,k,n) = Y(B,m,n) where C[b] differs for each b.

    Why Not Use Kronecker?
    ----------------------
    The Kronecker structure C.T ⊗ I_{B*m} (from _kron_nd_structure_rmul) assumes
    the SAME matrix C is applied to all batches. It creates a block-diagonal
    structure where C.T appears repeatedly.

    But with batch-varying constants, each batch needs its OWN coefficients:
        Y[b, i, j] = Σ_r X[b, i, r] * C[b, r, j]
                                      ↑
                                   Different C for each b!

    The Interleaved Structure for rmul
    ----------------------------------
    We need matrix A where vec(Y) = A @ vec(X) with:
        A[b + B*i + B*m*j, b' + B*i' + B*m*r] = C[b, r, j]  if b==b' and i==i'

    The key insight is the indexing pattern: instead of block-diagonal,
    we INTERLEAVE the batch dimension with the matrix dimensions.

    Vectorization (F-order):
        - vec(X) has index: b + B*i + B*m*r
        - vec(Y) has index: b + B*i + B*m*j

    Pattern: For each (b, r, j) in C and each row i:
        M[b + B*i + B*m*j, b + B*i + B*m*r] = C[b, r, j]

    Concrete Example
    ----------------
    C has shape (B=2, k=2, n=2):
        C[0] = [[a, b],      C[1] = [[e, f],
                [c, d]]              [g, h]]

    X has shape (B=2, m=1, k=2), so output Y has shape (B=2, m=1, n=2).

    Vectorization (F-order, m=1 so i dimension is trivial):
        vec(X) = [X[0,0,0], X[1,0,0], X[0,0,1], X[1,0,1]]
               = [x0,       x1,       x2,       x3      ]
                  └─b=0,1──┘ └──b=0,1──┘
                    r=0         r=1

        vec(Y) = [Y[0,0,0], Y[1,0,0], Y[0,0,1], Y[1,0,1]]
               = [y0,       y1,       y2,       y3      ]
                  └─b=0,1──┘ └──b=0,1──┘
                    j=0         j=1

    The computation:
        y0 = Y[0,0,0] = X[0,0,0]*C[0,0,0] + X[0,0,1]*C[0,1,0] = x0*a + x2*c
        y1 = Y[1,0,0] = X[1,0,0]*C[1,0,0] + X[1,0,1]*C[1,1,0] = x1*e + x3*g
        y2 = Y[0,0,1] = X[0,0,0]*C[0,0,1] + X[0,0,1]*C[0,1,1] = x0*b + x2*d
        y3 = Y[1,0,1] = X[1,0,0]*C[1,0,1] + X[1,0,1]*C[1,1,1] = x1*f + x3*h

    Interleaved matrix M (4×4):
                x0  x1  x2  x3
            ┌───────────────────┐
        y0  │  a   0   c   0   │   C[0,0,0]=a at (0,0), C[0,1,0]=c at (0,2)
        y1  │  0   e   0   g   │   C[1,0,0]=e at (1,1), C[1,1,0]=g at (1,3)
        y2  │  b   0   d   0   │   C[0,0,1]=b at (2,0), C[0,1,1]=d at (2,2)
        y3  │  0   f   0   h   │   C[1,0,1]=f at (3,1), C[1,1,1]=h at (3,3)
            └───────────────────┘

    Pattern: M[b + B*m*j, b + B*m*r] = C[b, r, j]
        - batch 0 elements (a,b,c,d) are at even rows/cols
        - batch 1 elements (e,f,g,h) are at odd rows/cols
        - No cross-batch interactions (zeros where b ≠ b')

    Contrast with Kronecker (if C were NOT batch-varying):
        Kronecker would put C.T in blocks: [[C.T,0],[0,C.T]]
        Interleaved disperses C's elements: rows/cols alternate by batch

    Parameters
    ----------
    const_data : np.ndarray
        Raw constant data (will be reshaped to (B, k, n) in Fortran order)
    const_shape : tuple
        Shape of the constant (..., k, n)
    var_shape : tuple
        Shape of the variable (..., m, k)

    Returns
    -------
    CooTensor
        The interleaved matrix for rmul
    """
    B = int(np.prod(const_shape[:-2]))
    k = const_shape[-2]
    n = const_shape[-1]
    m = var_shape[-2]

    # Reshape to (B, k, n) in Fortran order
    const_flat = np.reshape(const_data, (B, k, n), order="F")

    # Build interleaved indices for all (b, r, j) combinations
    b_idx = np.arange(B)
    r_idx = np.arange(k)
    j_idx = np.arange(n)

    bb, rr, jj = np.meshgrid(b_idx, r_idx, j_idx, indexing="ij")
    bb, rr, jj = bb.ravel(), rr.ravel(), jj.ravel()
    data = const_flat.ravel()

    # Base indices for i=0 case
    # Output: b + B*m*j, Input: b + B*m*r
    base_row = bb + B * m * jj
    base_col = bb + B * m * rr

    # Apply I_m: replicate for each i in 0..m-1
    if m > 1:
        i_offsets = np.arange(m)
        row_offsets = np.repeat(i_offsets * B, len(data))
        col_offsets = np.repeat(i_offsets * B, len(data))

        new_data = np.tile(data, m)
        new_row = np.tile(base_row, m) + row_offsets
        new_col = np.tile(base_col, m) + col_offsets
    else:
        new_data = data
        new_row = base_row
        new_col = base_col

    return CooTensor(
        data=new_data.astype(np.float64),
        row=new_row.astype(np.int64),
        col=new_col.astype(np.int64),
        param_idx=np.zeros(len(new_data), dtype=np.int64),
        m=B * m * n,
        n=B * m * k,
        param_size=1
    )


def coo_matmul(lhs: CooTensor, rhs: CooTensor) -> CooTensor:
    """
    Matrix multiplication of two CooTensors.

    lhs has shape (param_size_lhs, m, k) per slice
    rhs has shape (param_size_rhs, k, n) per slice

    For parametrized mul (A_param @ x):
    - lhs is the parameter tensor with param_size_lhs = param_size, shape (param_size, m, k)
    - rhs is the variable tensor with param_size_rhs = 1, shape (1, k, n)

    Result: (param_size_lhs, m, n) where each lhs slice is multiplied by the single rhs slice.

    This is O(nnz_lhs * nnz_per_row_rhs) instead of O(param_size * m * nnz_rhs).
    """
    if lhs.param_size > 1 and rhs.param_size == 1:
        # Common case: parametrized lhs @ constant rhs
        # For each non-zero in lhs at (p, i, j), multiply by rhs row j
        # Result goes to (p, i, :)

        # Build CSR-style indptr for rhs for fast row access (without scipy)
        rhs_indptr, rhs_sort_perm = compute_indptr(rhs.row, rhs.m)
        rhs_indices = rhs.col[rhs_sort_perm]  # col indices sorted by row
        rhs_data = rhs.data[rhs_sort_perm]

        # For each lhs entry, find how many rhs entries it will generate
        lhs_cols = lhs.col  # These index into rhs rows
        nnz_per_lhs = np.diff(rhs_indptr)[lhs_cols]  # nnz in each rhs row
        total_nnz = nnz_per_lhs.sum()

        if total_nnz == 0:
            return CooTensor.empty(lhs.m, rhs.n, lhs.param_size)

        # Expand lhs entries
        out_row = np.repeat(lhs.row, nnz_per_lhs)
        out_param = np.repeat(lhs.param_idx, nnz_per_lhs)

        # Get rhs data for each lhs entry
        rhs_starts = rhs_indptr[lhs_cols]

        # Build gather indices for rhs
        idx_offsets = np.arange(total_nnz) - np.repeat(
            np.concatenate([[0], np.cumsum(nnz_per_lhs)[:-1]]),
            nnz_per_lhs
        )
        gather_idx = np.repeat(rhs_starts, nnz_per_lhs) + idx_offsets

        out_col = rhs_indices[gather_idx]
        out_data = np.repeat(lhs.data, nnz_per_lhs) * rhs_data[gather_idx]

        return CooTensor(
            data=out_data,
            row=out_row,
            col=out_col,
            param_idx=out_param,
            m=lhs.m,
            n=rhs.n,
            param_size=lhs.param_size
        )

    elif lhs.param_size == 1 and rhs.param_size > 1:
        # Constant lhs @ parametrized rhs
        # For each rhs entry at (p, j, k), multiply by all lhs entries with col == j
        # Result goes to (p, lhs.row, k)

        # Build CSC-style indptr for lhs for fast column access (without scipy)
        lhs_indptr, lhs_sort_perm = compute_indptr(lhs.col, lhs.n)
        lhs_indices = lhs.row[lhs_sort_perm]  # row indices sorted by col
        lhs_data = lhs.data[lhs_sort_perm]

        # For each rhs entry, find how many lhs entries it will generate
        rhs_rows = rhs.row  # These index into lhs columns
        nnz_per_rhs = np.diff(lhs_indptr)[rhs_rows]  # nnz in each lhs column
        total_nnz = nnz_per_rhs.sum()

        if total_nnz == 0:
            return CooTensor.empty(lhs.m, rhs.n, rhs.param_size)

        # Expand rhs entries
        out_col = np.repeat(rhs.col, nnz_per_rhs)
        out_param = np.repeat(rhs.param_idx, nnz_per_rhs)

        # Get lhs data for each rhs entry
        lhs_starts = lhs_indptr[rhs_rows]

        # Build gather indices for lhs
        idx_offsets = np.arange(total_nnz) - np.repeat(
            np.concatenate([[0], np.cumsum(nnz_per_rhs)[:-1]]),
            nnz_per_rhs
        )
        gather_idx = np.repeat(lhs_starts, nnz_per_rhs) + idx_offsets

        out_row = lhs_indices[gather_idx]
        out_data = lhs_data[gather_idx] * np.repeat(rhs.data, nnz_per_rhs)

        return CooTensor(
            data=out_data,
            row=out_row,
            col=out_col,
            param_idx=out_param,
            m=lhs.m,
            n=rhs.n,
            param_size=rhs.param_size
        )

    elif lhs.param_size == 1 and rhs.param_size == 1:
        # Both constant - standard sparse matmul
        lhs_sparse = sp.csr_array(
            (lhs.data, (lhs.row, lhs.col)),
            shape=(lhs.m, lhs.n)
        )
        rhs_sparse = sp.csc_array(
            (rhs.data, (rhs.row, rhs.col)),
            shape=(rhs.m, rhs.n)
        )
        result = (lhs_sparse @ rhs_sparse).tocoo()

        return CooTensor(
            data=result.data.copy(),
            row=result.row.astype(np.int64),
            col=result.col.astype(np.int64),
            param_idx=np.zeros(len(result.data), dtype=np.int64),
            m=lhs.m,
            n=rhs.n,
            param_size=1
        )

    else:
        # Both operands parametrized - not supported in DPP
        raise ValueError(
            "coo_matmul: both operands have param_size > 1. "
            "This is not allowed in DPP-compliant problems."
        )


def coo_mul_elem(lhs: CooTensor, rhs: CooTensor) -> CooTensor:
    """
    Element-wise multiplication of two CooTensors.

    For parametrized case: lhs has param_size > 1, rhs has param_size = 1.
    Each lhs slice is multiplied element-wise by the single rhs slice.

    Handles broadcasting when rhs is a scalar (1x1).
    """
    if lhs.param_size > 1 and rhs.param_size == 1:
        # Check for scalar broadcast case: rhs is 1x1 constant
        if rhs.m == 1 and rhs.n == 1 and rhs.nnz > 0:
            # Scalar multiplication - multiply all lhs entries by rhs scalar
            scalar_val = rhs.data[0] if rhs.nnz == 1 else 0.0
            if scalar_val == 0:
                return CooTensor.empty(lhs.m, lhs.n, lhs.param_size)
            return CooTensor(
                data=lhs.data * scalar_val,
                row=lhs.row,
                col=lhs.col,
                param_idx=lhs.param_idx,
                m=lhs.m,
                n=lhs.n,
                param_size=lhs.param_size
            )

        # Handle case where lhs is in column format (n=1) but rhs has multiple columns
        # This happens with parametric mul_elem: param is (m, 1), var is (m, num_vars)
        # For element-wise multiply, match by row and use rhs columns
        if lhs.n == 1 and rhs.n > 1:
            # Vectorized: build CSR-style index on rhs by row
            rhs_indptr, rhs_sort_perm = compute_indptr(rhs.row, rhs.m)
            rhs_col_sorted = rhs.col[rhs_sort_perm]
            rhs_data_sorted = rhs.data[rhs_sort_perm]

            # For each lhs entry, count matching rhs entries by row
            lhs_rows = lhs.row
            nnz_per_lhs = np.diff(rhs_indptr)[lhs_rows]
            total_nnz = nnz_per_lhs.sum()

            if total_nnz == 0:
                return CooTensor.empty(lhs.m, rhs.n, lhs.param_size)

            # Expand lhs entries
            out_row = np.repeat(lhs.row, nnz_per_lhs)
            out_param = np.repeat(lhs.param_idx, nnz_per_lhs)
            lhs_vals = np.repeat(lhs.data, nnz_per_lhs)

            # Build gather indices for rhs (same pattern as coo_matmul)
            rhs_starts = rhs_indptr[lhs_rows]
            idx_offsets = np.arange(total_nnz) - np.repeat(
                np.concatenate([[0], np.cumsum(nnz_per_lhs)[:-1]]),
                nnz_per_lhs
            )
            gather_idx = np.repeat(rhs_starts, nnz_per_lhs) + idx_offsets

            out_col = rhs_col_sorted[gather_idx]
            out_data = lhs_vals * rhs_data_sorted[gather_idx]

            return CooTensor(
                data=out_data,
                row=out_row,
                col=out_col,
                param_idx=out_param,
                m=lhs.m,
                n=rhs.n,
                param_size=lhs.param_size
            )

        # Handle case where lhs has multiple columns but rhs is column vector
        # This happens in div: lhs is param view (m, n), rhs is column vector (m, 1)
        # For element-wise multiply, broadcast rhs across all lhs columns
        if lhs.n > 1 and rhs.n == 1:
            # Build CSR-style index on rhs by row for efficient lookup
            rhs_indptr, rhs_sort_perm = compute_indptr(rhs.row, rhs.m)
            rhs_data_sorted = rhs.data[rhs_sort_perm]

            # Vectorized: for each lhs entry, check if rhs has an entry at same row
            lhs_rows = lhs.row
            rhs_starts = rhs_indptr[lhs_rows]
            rhs_ends = rhs_indptr[lhs_rows + 1]
            has_match = rhs_ends > rhs_starts  # Boolean mask

            if not np.any(has_match):
                return CooTensor.empty(lhs.m, lhs.n, lhs.param_size)

            # Filter to only entries with matching rhs values
            out_data = lhs.data[has_match] * rhs_data_sorted[rhs_starts[has_match]]
            out_row = lhs.row[has_match]
            out_col = lhs.col[has_match]
            out_param = lhs.param_idx[has_match]

            return CooTensor(
                data=out_data,
                row=out_row,
                col=out_col,
                param_idx=out_param,
                m=lhs.m,
                n=lhs.n,
                param_size=lhs.param_size
            )

        # General case: rhs has same dimensions as lhs slices
        # Use sparse lookup to avoid allocating large dense arrays
        # Create a sparse CSR matrix from rhs for efficient lookup
        rhs_sparse = sp.csr_array(
            (rhs.data, (rhs.row, rhs.col)),
            shape=(rhs.m, rhs.n)
        )

        # For each lhs entry, get the corresponding rhs value
        # Validate that lhs indices are within rhs bounds
        if lhs.data.size > 0:
            if np.any(lhs.row >= rhs.m) or np.any(lhs.col >= rhs.n):
                raise ValueError(
                    f"Index out of bounds in mul_elem: lhs indices must be within "
                    f"rhs shape ({rhs.m}, {rhs.n})"
                )

        # Get rhs values at lhs positions using sparse indexing
        rhs_vals = np.asarray(rhs_sparse[lhs.row, lhs.col]).ravel()

        # Keep only non-zero results
        mask = rhs_vals != 0
        return CooTensor(
            data=lhs.data[mask] * rhs_vals[mask],
            row=lhs.row[mask],
            col=lhs.col[mask],
            param_idx=lhs.param_idx[mask],
            m=lhs.m,
            n=lhs.n,
            param_size=lhs.param_size
        )

    elif lhs.param_size == 1 and rhs.param_size > 1:
        # Swap and recurse
        return coo_mul_elem(rhs, lhs)

    elif lhs.param_size == 1 and rhs.param_size == 1:
        # Both constant - standard sparse element-wise product
        lhs_csr = lhs.to_stacked_sparse()
        rhs_csr = rhs.to_stacked_sparse()
        result = lhs_csr.multiply(rhs_csr).tocoo()
        return CooTensor(
            data=result.data.copy(),
            row=result.row.copy(),
            col=result.col.copy(),
            param_idx=np.zeros(len(result.data), dtype=np.int64),
            m=lhs.m,
            n=lhs.n,
            param_size=1
        )

    else:
        # Both operands parametrized - not supported in DPP
        raise ValueError(
            "coo_mul_elem: both operands have param_size > 1. "
            "This is not allowed in DPP-compliant problems."
        )


def coo_reshape(tensor: CooTensor, new_m: int, new_n: int) -> CooTensor:
    """
    Reshape the tensor (Fortran order, column-major).

    For each entry at (row, col), compute linear index = col * m + row,
    then new_row = linear_idx % new_m, new_col = linear_idx // new_m.

    The param_idx is preserved - it identifies which parameter value affects
    each entry, not the position in the matrix.

    This function is used by the `reshape` linop for general reshape operations.
    For reshaping parametric constant data (from get_constant_data), use
    reshape_parametric_constant instead.
    """
    # Compute linear index in column-major order
    linear_idx = tensor.col * tensor.m + tensor.row

    # Compute new row and col
    new_row = linear_idx % new_m
    new_col = linear_idx // new_m

    return CooTensor(
        data=tensor.data.copy(),
        row=new_row.astype(np.int64),
        col=new_col.astype(np.int64),
        param_idx=tensor.param_idx.copy(),
        m=new_m,
        n=new_n,
        param_size=tensor.param_size
    )


def reshape_parametric_constant(tensor: CooTensor, new_m: int, new_n: int) -> CooTensor:
    """
    Reshape parametric constant data from column format to matrix format,
    with deduplication of broadcast entries.

    This function is used when extracting constant data (via get_constant_data)
    that needs to be reshaped from column vector format to matrix format for
    operations like matrix multiplication.

    Why Deduplication?
    ------------------
    When a parameter is broadcast to a larger shape (e.g., P(2,3) → P(4,2,3)),
    the `select_rows` operation duplicates entries, including their param_idx
    values. See `_select_rows_with_duplicates` for details.

    After broadcast, the same param_idx appears multiple times:
        param_idx = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
                     └─4 copies─┘ └─4 copies─┘

    Without deduplication, when we reshape to matrix format:
        - We'd create 4 copies of each matrix position
        - The resulting matrix would have duplicate entries at same (row, col)
        - This wastes memory and can cause incorrect results when the sparse
          matrix sums duplicate entries

    The fix: keep only the FIRST occurrence of each param_idx value.

    Concrete Example
    ----------------
    Parameter P has shape (2, 3), broadcast to (4, 2, 3) for batch matmul.

    Original P (before broadcast):
        param_idx = [0, 1, 2, 3, 4, 5]  (6 unique elements)
        data = [1, 1, 1, 1, 1, 1]       (coefficients, all 1s)
        row = [0, 1, 2, 3, 4, 5]        (column vector format)

    After broadcast via select_rows (24 entries):
        param_idx = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5]
        data = [1,1,1,1, 1,1,1,1, ...]  (24 entries)
        row = [0,2,4,6, 1,3,5,7, ...]   (spread across output rows)

    After reshape_parametric_constant to (2, 3):
        param_idx = [0, 1, 2, 3, 4, 5]  (deduplicated!)
        data = [1, 1, 1, 1, 1, 1]       (first occurrence of each)
        row = [0, 1, 0, 1, 0, 1]        (param_idx % 2)
        col = [0, 0, 1, 1, 2, 2]        (param_idx // 2)

    The result is a proper (2, 3) sparse matrix with one entry per position,
    ready for matrix multiplication.

    Position Calculation
    --------------------
    For parametric tensors, param_idx directly encodes the position in the
    parameter matrix (column-major/Fortran order):
        new_row = param_idx % new_m
        new_col = param_idx // new_m

    For example, param_idx=3 in a (2, 3) matrix:
        row = 3 % 2 = 1
        col = 3 // 2 = 1
        → position (1, 1) in the matrix

    Parameters
    ----------
    tensor : CooTensor
        Input tensor, typically in column format from get_constant_data
    new_m : int
        Number of rows in output matrix
    new_n : int
        Number of columns in output matrix

    Returns
    -------
    CooTensor
        Reshaped tensor with deduplicated entries (for parametric case)
    """
    if tensor.param_size > 1:
        # For parametric tensors, use param_idx as position indicator
        # Each param_idx maps to one position in (new_m, new_n) in column-major order
        new_row = tensor.param_idx % new_m
        new_col = tensor.param_idx // new_m

        # Deduplicate: keep first occurrence of each param_idx
        # This handles broadcast operations that duplicate param entries
        unique_param_idx, first_occurrence = np.unique(
            tensor.param_idx, return_index=True
        )

        return CooTensor(
            data=tensor.data[first_occurrence].copy(),
            row=new_row[first_occurrence].astype(np.int64),
            col=new_col[first_occurrence].astype(np.int64),
            param_idx=unique_param_idx.astype(np.int64),
            m=new_m,
            n=new_n,
            param_size=tensor.param_size
        )
    else:
        # Non-parametric: use standard linear index reshaping
        return coo_reshape(tensor, new_m, new_n)


class CooTensorView(DictTensorView):
    """
    TensorView using CooTensor storage for O(nnz) operations.

    Unlike SciPyTensorView which stores stacked sparse matrices of shape
    (param_size * m, n), CooTensorView stores CooTensor objects that
    keep (data, row, col, param_idx) separately.

    This avoids O(param_size * m) operations, making it much faster for
    large parameter matrices.
    """

    @property
    def rows(self) -> int:
        """Number of rows of the TensorView (per parameter slice)."""
        if self.tensor is not None:
            for param_dict in self.tensor.values():
                for param_id, compact in param_dict.items():
                    return compact.m
        else:
            raise ValueError('Tensor cannot be None')

    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.

        This is trivial for CooTensor - just add offsets.
        """
        assert self.tensor is not None
        shape = (total_rows, self.var_length + 1)
        tensor_representations = []

        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, compact in variable_tensor.items():
                tensor_representations.append(TensorRepresentation(
                    compact.data,
                    compact.row + row_offset,
                    compact.col + self.id_to_col[variable_id],
                    compact.param_idx + self.param_to_col[parameter_id],
                    shape=shape
                ))

        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from each parameter slice.

        O(nnz) operation - just filter entries by row membership.
        """
        def func(compact, p):
            return compact.select_rows(rows)

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.

        func signature: func(compact: CooTensor, p: int) -> CooTensor
        """
        self.tensor = {var_id: {k: func(v, self.param_to_size[k])
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> 'CooTensorView':
        """Create new CooTensorView with same shape information."""
        return CooTensorView(
            variable_ids, tensor, is_parameter_free,
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, CooTensor]) \
            -> dict[int, CooTensor]:
        """Apply 'func' to each parameter slice."""
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: CooTensor, b: CooTensor) -> CooTensor:
        """Add two CooTensors."""
        return a + b

    @staticmethod
    def tensor_type():
        """The tensor type for CooTensorView."""
        return CooTensor


class CooCanonBackend(PythonCanonBackend):
    """
    Canon backend using CooTensorView for O(nnz) operations.

    This backend stores tensors in compact COO format with separate
    parameter indices, avoiding the creation of huge stacked matrices.
    """

    def get_empty_view(self) -> CooTensorView:
        """Return an empty CooTensorView."""
        return CooTensorView.get_empty_view(
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def get_variable_tensor(self, shape: tuple, var_id: int) -> dict:
        """
        Create tensor for a variable.

        Returns {var_id: {Constant.ID: tensor}} where tensor is identity-like.
        """
        size = int(np.prod(shape))
        compact = CooTensor(
            data=np.ones(size, dtype=np.float64),
            row=np.arange(size, dtype=np.int64),
            col=np.arange(size, dtype=np.int64),
            param_idx=np.zeros(size, dtype=np.int64),
            m=size,
            n=size,
            param_size=1
        )
        return {var_id: {Constant.ID.value: compact}}

    def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> dict:
        """
        Create tensor for constant data.

        Returns {Constant.ID: {Constant.ID: tensor}} as column vector.
        """
        if sp.issparse(data):
            # Extract directly from sparse format (avoid densification)
            coo = data.tocoo()
            size = coo.shape[0] * coo.shape[1]
            # Flatten indices in Fortran order: linear_idx = col * nrows + row
            nz_idx = coo.col * coo.shape[0] + coo.row
            nz_data = coo.data.copy()
        else:
            flat = np.asarray(data).flatten(order='F')
            size = len(flat)
            # Find non-zero entries
            nz_mask = flat != 0
            nz_idx = np.where(nz_mask)[0]
            nz_data = flat[nz_mask]

        compact = CooTensor(
            data=nz_data,
            row=nz_idx.astype(np.int64),
            col=np.zeros(len(nz_idx), dtype=np.int64),
            param_idx=np.zeros(len(nz_idx), dtype=np.int64),
            m=size,
            n=1,
            param_size=1
        )
        return {Constant.ID.value: {Constant.ID.value: compact}}

    def get_param_tensor(self, shape: tuple, parameter_id: int) -> dict:
        """
        Create tensor for a parameter.

        Returns {Constant.ID: {parameter_id: tensor}}.
        Each parameter element becomes a separate slice in the 3D tensor.
        """
        param_size = self.param_to_size[parameter_id]
        size = int(np.prod(shape))

        # Each parameter element gets its own slice
        # param[i] = 1 at position (i, 0) in slice i
        compact = CooTensor(
            data=np.ones(param_size, dtype=np.float64),
            row=np.arange(param_size, dtype=np.int64),
            col=np.zeros(param_size, dtype=np.int64),
            param_idx=np.arange(param_size, dtype=np.int64),
            m=size,
            n=1,
            param_size=param_size
        )
        return {Constant.ID.value: {parameter_id: compact}}

    def concatenate_tensors(self, tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """Combine multiple tensor representations."""
        return TensorRepresentation.combine(tensors)

    # =========================================================================
    # Tensor operations
    # =========================================================================

    def sum_entries(self, lin_op, view: CooTensorView) -> CooTensorView:
        """Sum entries along an axis (ND-aware)."""
        shape = tuple(lin_op.args[0].shape)

        # Handle None data (from trace or simple sum)
        if lin_op.data is None:
            axis = None
        else:
            axis, _ = lin_op.data

        if axis is None:
            # Sum all entries to scalar
            def func(compact, p):
                return CooTensor(
                    data=compact.data.copy(),
                    row=np.zeros(compact.nnz, dtype=np.int64),
                    col=compact.col.copy(),
                    param_idx=compact.param_idx.copy(),
                    m=1,
                    n=compact.n,
                    param_size=compact.param_size
                )
            view.accumulate_over_variables(func, is_param_free_function=True)
        else:
            # Sum along specific axis
            # row_map[i] tells us which output row input row i maps to
            row_map = self._get_sum_row_map(shape, axis)
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            out_axes = [i for i in range(len(shape)) if i not in axis_tuple]
            new_m = int(np.prod([shape[i] for i in out_axes])) if out_axes else 1

            def func(compact, p):
                return CooTensor(
                    data=compact.data.copy(),
                    row=row_map[compact.row],  # Map each entry's row to output row
                    col=compact.col.copy(),
                    param_idx=compact.param_idx.copy(),
                    m=new_m,
                    n=compact.n,
                    param_size=compact.param_size
                )
            view.accumulate_over_variables(func, is_param_free_function=True)

        return view

    def _get_sum_row_map(self, shape: tuple, axis) -> np.ndarray:
        """
        Compute row mapping for axis-specific sum.

        Returns array where row_map[i] is the output row for input row i.
        """
        axis = axis if isinstance(axis, tuple) else (axis,)
        out_axes = np.isin(range(len(shape)), axis, invert=True)
        out_idx = np.indices(shape)[out_axes]
        out_dims = np.array(shape)[out_axes]
        row_idx = np.ravel_multi_index(out_idx, dims=out_dims, order='F')
        return row_idx.flatten(order='F')

    def reshape(self, lin_op, view: CooTensorView) -> CooTensorView:
        """Reshape tensor (column-major/Fortran order).

        Note: C-order reshape is handled at the atom level by transposing
        before and after the reshape operation (see reshape.graph_implementation).
        The backend reshape always uses F-order.
        """
        new_shape = lin_op.shape
        new_m = int(np.prod(new_shape))

        def func(compact, p):
            return coo_reshape(compact, new_m, compact.n)

        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    # transpose: use base class implementation (via select_rows)

    def _mul_kronecker(
        self,
        lhs_data,
        const_shape: tuple,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        Case 3: 2D constant @ ND variable using Kronecker structure.

        The SAME constant matrix C is applied to all batch elements.
        Uses I_n ⊗ C ⊗ I_B structure (see _kron_nd_structure_mul for details).

        Example: C(2,3) @ X(4,3,5) → Y(4,2,5)
            Same 2×3 matrix applied to all 4 batch elements.
        """
        batch_size, n, _ = get_nd_matmul_dims(const_shape, var_shape)

        # Convert to CooTensor if needed
        if isinstance(lhs_data, CooTensor):
            lhs_tensor = lhs_data
        else:
            lhs_tensor = self._to_coo_tensor(lhs_data)

        stacked_compact = _kron_nd_structure_mul(lhs_tensor, batch_size, n)

        def constant_mul(compact, p):
            return coo_matmul(stacked_compact, compact)

        view.accumulate_over_variables(constant_mul, is_param_free_function=True)
        return view

    def _mul_interleaved(
        self,
        lin_op,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        Case 2: Batch-varying constant @ variable using interleaved structure.

        Each batch element uses a DIFFERENT constant matrix.
        Cannot use Kronecker (which assumes same matrix for all batches).
        Uses interleaved indexing (see _build_interleaved_mul for details).

        Example: C(4,2,3) @ X(4,3,5) → Y(4,2,5)
            Four different 2×3 matrices, one per batch element.
        """
        const_shape = lin_op.data.shape
        # Raw data access is intentional: batch-varying constants are never parametric.
        # lin_op.data is a LinOp of type "*_const", so lin_op.data.data gets the numpy array.
        assert lin_op.data.type in {"dense_const", "sparse_const", "scalar_const"}, \
            "Batch-varying constants must be non-parametric"
        stacked_compact = _build_interleaved_mul(lin_op.data.data, const_shape, var_shape)

        def constant_mul(compact, p):
            return coo_matmul(stacked_compact, compact)

        view.accumulate_over_variables(constant_mul, is_param_free_function=True)
        return view

    def _mul_parametric_lhs(
        self,
        lhs_data: dict,
        const_shape: tuple,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        Case 1: Parametric constant @ variable.

        The constant is a cp.Parameter - values unknown at canonicalization.
        Each parameter element gets its own slice in the 3D tensor, tracking
        how that element affects each output position.

        Uses Kronecker structure per parameter slice (same as Case 3, but
        applied to each param_idx separately).

        Example: P(2,3) @ X(4,3,5) → Y(4,2,5) where P is cp.Parameter((2,3))
            6 parameter elements, each with its own contribution matrix.
        """
        batch_size, n, _ = get_nd_matmul_dims(const_shape, var_shape)

        # Expand each parameter slice with Kronecker structure
        expanded_lhs = {
            param_id: _kron_nd_structure_mul(tensor, batch_size, n)
            for param_id, tensor in lhs_data.items()
        }

        def parametrized_mul(rhs_compact):
            return {
                param_id: coo_matmul(lhs_compact, rhs_compact)
                for param_id, lhs_compact in expanded_lhs.items()
            }

        # Apply to each variable tensor in-place
        for var_id, var_tensor in view.tensor.items():
            const_compact = var_tensor[Constant.ID.value]
            view.tensor[var_id] = parametrized_mul(const_compact)

        view.is_parameter_free = False
        return view

    def mul(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Matrix multiply: C @ X where C is constant/parametric, X is variable.

        Three Cases
        -----------
        The constant C can be in one of three forms, each requiring different handling:

        1. **Parametric** (`_mul_parametric_lhs`):
           C is a `cp.Parameter`. Values unknown at canonicalization time.
           - Example: P(2,3) @ X(3,4) where P is cp.Parameter((2,3))
           - Each parameter element gets its own slice in the 3D tensor
           - Uses Kronecker structure per parameter slice

        2. **Batch-varying constant** (`_mul_interleaved`):
           C has shape (B,m,k) with B > 1. Different matrix for each batch.
           - Example: C(4,2,3) @ X(4,3,5) - four different 2×3 matrices
           - Cannot use Kronecker (that assumes SAME matrix for all batches)
           - Uses interleaved indexing: M[b + B*i, b + B*r] = C[b,i,r]

        3. **2D constant** (`_mul_kronecker`):
           C has shape (m,k) or (1,m,k). Same matrix for all batches.
           - Example: C(2,3) @ X(4,3,5) - same 2×3 matrix applied 4 times
           - Uses Kronecker structure: I_n ⊗ C ⊗ I_B (see _kron_nd_structure_mul)

        Why Three Cases?
        ----------------
        - Case 1 vs 2,3: Parametric is fundamentally different because we don't
          know values at canonicalization. We track param_idx to know which
          parameter element affects which output.

        - Case 2 vs 3: Both are known constants, but the matrix structure differs:
          - Case 3: Same C repeated → block-diagonal (Kronecker)
          - Case 2: Different C per batch → interleaved positions

        Detection Logic
        ---------------
        - Case 1: hasattr(const, 'type') and const.type == 'param'
        - Case 2: len(const_shape) == 3 and const_shape[0] > 1  (batch-varying)
        - Case 3: Otherwise (2D or trivial batch dim)
        """
        const = lin_op.data
        var_shape = lin_op.args[0].shape
        const_shape = const.shape

        # Get constant data - check for direct param case first
        if hasattr(const, 'type') and const.type == 'param':
            # Direct parameter: construct CooTensor for parameter matrix
            # Parameters are stored in column-major (Fortran) order:
            # param_idx=0 -> A[0,0], param_idx=1 -> A[1,0], ..., param_idx=m -> A[0,1]
            param_id = const.data
            param_size = self.param_to_size[param_id]
            size = int(np.prod(const.shape))
            m, k = const.shape if len(const.shape) == 2 else (1, size)
            lhs_data = {param_id: CooTensor(
                data=np.ones(param_size, dtype=np.float64),
                row=np.tile(np.arange(m), k),
                col=np.repeat(np.arange(k), m),
                param_idx=np.arange(param_size, dtype=np.int64),
                m=m, n=k, param_size=param_size
            )}
            is_param_free = False
        else:
            # Compute 2D target shape (last 2 dims for ND, row vector for 1D)
            target = const_shape[-2:] if len(const_shape) >= 2 else (1, const_shape[0])
            lhs_data, is_param_free = self.get_constant_data(const, view, target_shape=target)

        if not is_param_free:
            # Case 1: Parametric LHS
            return self._mul_parametric_lhs(lhs_data, const_shape, var_shape, view)
        elif is_batch_varying(const_shape):
            # Case 2: Batch-varying constant
            return self._mul_interleaved(lin_op, var_shape, view)
        else:
            # Case 3: 2D constant (or trivial batch dims like (1, m, k))
            return self._mul_kronecker(lhs_data, const_shape, var_shape, view)

    def get_constant_data_from_const(self, lin_op):
        """Extract constant data from a LinOp."""
        if lin_op.type == 'scalar_const':
            return np.array([[lin_op.data]])
        elif lin_op.type == 'dense_const':
            return lin_op.data
        elif lin_op.type == 'sparse_const':
            return lin_op.data
        else:
            raise ValueError(f"Unknown const type: {lin_op.type}")

    def div(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Division by constant: x / d.

        Note: div currently doesn't support parameters in divisor.
        """
        rhs, is_param_free_rhs = self.get_constant_data(lin_op.data, view, target_shape=None)
        assert is_param_free_rhs, "div doesn't support parametrized divisor"

        # Get reciprocal values
        rhs_compact = self._to_coo_tensor(rhs)
        # Check for zero divisors (both explicit zeros and implicit zeros from sparsity)
        if np.any(rhs_compact.data == 0):
            raise ValueError("Division by zero encountered in divisor (explicit zero)")
        expected_entries = rhs_compact.m * rhs_compact.n * rhs_compact.param_size
        if rhs_compact.nnz < expected_entries:
            raise ValueError("Division by zero encountered in divisor (sparse with implicit zeros)")
        # Invert the data
        recip_data = np.reciprocal(rhs_compact.data, dtype=float)
        rhs_recip = CooTensor(
            data=recip_data,
            row=rhs_compact.row,
            col=rhs_compact.col,
            param_idx=rhs_compact.param_idx,
            m=rhs_compact.m,
            n=rhs_compact.n,
            param_size=rhs_compact.param_size
        )

        def div_func(compact, p):
            return coo_mul_elem(compact, rhs_recip)

        view.accumulate_over_variables(div_func, is_param_free_function=True)
        return view

    def mul_elem(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Element-wise multiplication: x * d.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, target_shape=None)

        if is_param_free_lhs:
            lhs_compact = self._to_coo_tensor(lhs)

            def func(compact, p):
                return coo_mul_elem(compact, lhs_compact)

            view.accumulate_over_variables(func, is_param_free_function=True)
        else:
            # Parametrized mul_elem
            def parametrized_mul_elem(rhs_compact):
                result = {}
                for param_id, lhs_compact in lhs.items():
                    lhs_ct = self._to_coo_tensor(lhs_compact, param_id)
                    result[param_id] = coo_mul_elem(lhs_ct, rhs_compact)
                return result

            # Apply to each variable tensor in-place
            for var_id, var_tensor in view.tensor.items():
                const_compact = var_tensor[Constant.ID.value]
                view.tensor[var_id] = parametrized_mul_elem(const_compact)

            view.is_parameter_free = False
            return view

        return view

    @staticmethod
    def promote(lin_op, view: CooTensorView) -> CooTensorView:
        """
        Promote a scalar to a higher-dimensional shape by repeating.

        Creates rows = [0, 0, 0, ...] (all zeros), which causes the single
        scalar entry to be replicated to all output positions via
        _select_rows_with_duplicates.

        For a parametric scalar, this means the same param_idx=0 entry
        will appear multiple times in the output, once for each position
        in the target shape.
        """
        num_entries = int(np.prod(lin_op.shape))
        # All zeros = copy the single row (scalar) to every output position
        rows = np.zeros(num_entries, dtype=np.int64)
        view.select_rows(rows)
        return view

    @staticmethod
    def broadcast_to(lin_op, view: CooTensorView) -> CooTensorView:
        """
        Broadcast tensor to a larger shape by replicating elements.

        This operation creates duplicate row indices, which causes the same
        tensor entries (including their param_idx values) to be replicated
        to multiple output positions via _select_rows_with_duplicates.

        Example: Parameter P of shape (2, 3) broadcast to (4, 2, 3)
        -----------------------------------------------------------
        Original row indices (column-major flattening of (2,3)):
            [[0, 2, 4],
             [1, 3, 5]]

        After np.broadcast_to to (4, 2, 3) and flatten:
            rows = [0, 1, 0, 1, 0, 1, 0, 1,  # batch dim varies fastest
                    2, 3, 2, 3, 2, 3, 2, 3,  # then original rows
                    4, 5, 4, 5, 4, 5, 4, 5]  # 24 total elements

        Each original index appears 4 times (once per batch).

        The select_rows call will replicate tensor entries accordingly,
        so a parameter element at row 0 will contribute to output rows
        0, 2, 4, 6 (all positions where original row 0 appears).

        For parametric tensors, this means the same param_idx appears
        multiple times in the output - which is semantically correct
        because the same parameter value is used in all broadcast copies.
        """
        broadcast_shape = lin_op.shape
        original_shape = lin_op.args[0].shape
        # Create indices for original shape elements
        rows = np.arange(np.prod(original_shape, dtype=int)).reshape(original_shape, order='F')
        # Broadcast creates duplicates: same original index appears multiple times
        rows = np.broadcast_to(rows, broadcast_shape).flatten(order="F")
        # select_rows handles the duplication via _select_rows_with_duplicates
        view.select_rows(rows.astype(np.int64))
        return view

    # index: use base class implementation (via select_rows)

    @staticmethod
    def diag_vec(lin_op, view: CooTensorView) -> CooTensorView:
        """Convert vector to diagonal matrix."""
        k = lin_op.data  # Diagonal offset
        n = lin_op.shape[0]
        total_rows = n * n

        def func(compact, p):
            # Map each entry to diagonal position
            if k == 0:
                new_row = compact.row * (n + 1)
            elif k > 0:
                new_row = compact.row * (n + 1) + n * k
            else:
                new_row = compact.row * (n + 1) - k

            return CooTensor(
                data=compact.data.copy(),
                row=new_row.astype(np.int64),
                col=compact.col.copy(),
                param_idx=compact.param_idx.copy(),
                m=total_rows,
                n=compact.n,
                param_size=compact.param_size
            )

        view.apply_all(func)
        return view

    # diag_mat: use base class implementation (via select_rows)
    # trace: use base class implementation (via select_rows)
    # upper_tri: use base class implementation (via select_rows)

    def _to_coo_tensor(self, matrix, param_id=None):
        """Convert sparse matrix or dense array to CooTensor."""
        if isinstance(matrix, dict):
            raise ValueError("Expected single matrix, got dict")

        if isinstance(matrix, np.ndarray):
            # Dense array: extract nonzeros directly
            data_2d = np.atleast_2d(matrix)
            rows, cols = np.nonzero(data_2d)
            data = data_2d[rows, cols]
            m, n = data_2d.shape
        else:
            # Sparse matrix
            coo = matrix.tocoo()
            rows, cols, data = coo.row, coo.col, coo.data.copy()
            m, n = coo.shape

        if param_id is not None:
            p = self.param_to_size[param_id]
            slice_m = m // p
            param_idx, rows = np.divmod(rows, slice_m)
            m = slice_m
        else:
            p = 1
            param_idx = np.zeros(len(data), dtype=np.int64)

        return CooTensor(
            data=data.astype(np.float64),
            row=rows.astype(np.int64),
            col=cols.astype(np.int64),
            param_idx=param_idx.astype(np.int64),
            m=m,
            n=n,
            param_size=p
        )

    # vstack, hstack, concatenate: use base class implementations

    def _rmul_kronecker(
        self,
        rhs_data,
        const_shape: tuple,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        ND rmul Case 3: ND variable @ 2D constant using Kronecker structure.

        The SAME constant matrix C is applied to all batch elements.
        Uses C.T tensor I_{B*m} structure.

        Example: X(4,3,5) @ C(5,2) -> Y(4,3,2)
            Same 5x2 matrix applied to all 4 batch elements.
        """
        batch_size, m, n, _ = get_nd_rmul_dims(var_shape, const_shape)

        # Convert to CooTensor if needed
        if isinstance(rhs_data, CooTensor):
            rhs_tensor = rhs_data
        else:
            rhs_tensor = self._to_coo_tensor(rhs_data)

        # Apply C.T tensor I_{B*m}
        stacked_compact = _kron_nd_structure_rmul(rhs_tensor, batch_size, m)

        def constant_rmul(compact, p):
            return coo_matmul(stacked_compact, compact)

        view.accumulate_over_variables(constant_rmul, is_param_free_function=True)
        return view

    def _rmul_interleaved(
        self,
        lin_op,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        ND rmul Case 2: Batch-varying rmul using interleaved structure.

        Each batch element uses a DIFFERENT constant matrix.

        Example: X(4,3,5) @ C(4,5,2) -> Y(4,3,2)
            Four different 5x2 matrices, one per batch element.
        """
        const_shape = lin_op.data.shape
        assert lin_op.data.type in {"dense_const", "sparse_const", "scalar_const"}, \
            "Batch-varying constants must be non-parametric"
        stacked_compact = _build_interleaved_rmul(lin_op.data.data, const_shape, var_shape)

        def constant_rmul(compact, p):
            return coo_matmul(stacked_compact, compact)

        view.accumulate_over_variables(constant_rmul, is_param_free_function=True)
        return view

    def _rmul_parametric_rhs(
        self,
        rhs_data: dict,
        const_shape: tuple,
        var_shape: tuple,
        view: CooTensorView,
    ) -> CooTensorView:
        """
        ND rmul Case 1: Variable @ parametric constant.

        The constant is a cp.Parameter - values unknown at canonicalization.
        Each parameter element gets its own slice in the 3D tensor.

        Uses Kronecker structure per parameter slice.

        Example: X(4,3,5) @ P(5,2) -> Y(4,3,2) where P is cp.Parameter((5,2))
            10 parameter elements, each with its own contribution matrix.
        """
        batch_size, m, n, _ = get_nd_rmul_dims(var_shape, const_shape)

        # Expand each parameter slice with Kronecker structure for rmul
        expanded_rhs = {
            param_id: _kron_nd_structure_rmul(tensor, batch_size, m)
            for param_id, tensor in rhs_data.items()
        }

        def parametrized_rmul(lhs_compact):
            return {
                param_id: coo_matmul(rhs_compact, lhs_compact)
                for param_id, rhs_compact in expanded_rhs.items()
            }

        # Apply to each variable tensor in-place
        for var_id, var_tensor in view.tensor.items():
            const_compact = var_tensor[Constant.ID.value]
            view.tensor[var_id] = parametrized_rmul(const_compact)

        view.is_parameter_free = False
        return view

    def rmul(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Right multiplication: X @ C where X is variable, C is constant/parametric.

        For X @ C where X is (m, k) variable and C is (k, n) constant:
        vec(X @ C) = (C.T tensor I_m) @ vec(X)

        Three Cases
        -----------
        The constant C can be in one of three forms, each requiring different handling:

        1. **Parametric** (`_rmul_parametric_rhs`):
           C is a `cp.Parameter`. Values unknown at canonicalization time.
           - Example: X(3,4) @ P(4,2) where P is cp.Parameter((4,2))
           - Each parameter element gets its own slice in the 3D tensor
           - Uses Kronecker structure per parameter slice

        2. **Batch-varying constant** (`_rmul_interleaved`):
           C has shape (B,k,n) with B > 1. Different matrix for each batch.
           - Example: X(4,3,5) @ C(4,5,2) - four different 5×2 matrices
           - Cannot use Kronecker (that assumes SAME matrix for all batches)
           - Uses interleaved indexing

        3. **2D constant** (`_rmul_kronecker`):
           C has shape (k,n) or (1,k,n). Same matrix for all batches.
           - Example: X(4,3,5) @ C(5,2) - same 5×2 matrix applied 4 times
           - Uses Kronecker structure: C.T ⊗ I_{B*m} (see _kron_nd_structure_rmul)

        Why Three Cases?
        ----------------
        - Case 1 vs 2,3: Parametric is fundamentally different because we don't
          know values at canonicalization. We track param_idx to know which
          parameter element affects which output.

        - Case 2 vs 3: Both are known constants, but the matrix structure differs:
          - Case 3: Same C repeated → Kronecker structure
          - Case 2: Different C per batch → interleaved positions

        Detection Logic
        ---------------
        - Case 1: hasattr(const, 'type') and const.type == 'param'
        - Case 2: len(const_shape) == 3 and const_shape[0] > 1  (batch-varying)
        - Case 3: Otherwise (2D or trivial batch dim)
        """
        const = lin_op.data
        var_shape = lin_op.args[0].shape
        const_shape = const.shape

        # Get constant data - check for direct param case first
        if hasattr(const, 'type') and const.type == 'param':
            # Direct parameter: construct CooTensor for parameter matrix
            # Parameters are stored in column-major (Fortran) order:
            # param_idx=0 -> C[0,0], param_idx=1 -> C[1,0], ..., param_idx=k -> C[0,1]
            param_id = const.data
            param_size = self.param_to_size[param_id]
            size = int(np.prod(const.shape))
            k, n = const.shape if len(const.shape) == 2 else (size, 1)
            rhs_data = {param_id: CooTensor(
                data=np.ones(param_size, dtype=np.float64),
                row=np.tile(np.arange(k), n),
                col=np.repeat(np.arange(n), k),
                param_idx=np.arange(param_size, dtype=np.int64),
                m=k, n=n, param_size=param_size
            )}
            is_param_free = False
        else:
            # Compute 2D target shape (last 2 dims for ND, column vector for 1D)
            target = const_shape[-2:] if len(const_shape) >= 2 else (const_shape[0], 1)
            rhs_data, is_param_free = self.get_constant_data(const, view, target_shape=target)

        if not is_param_free:
            # Case 1: Parametric RHS
            return self._rmul_parametric_rhs(rhs_data, const_shape, var_shape, view)
        elif is_batch_varying(const_shape):
            # Case 2: Batch-varying constant
            return self._rmul_interleaved(lin_op, var_shape, view)
        else:
            # Case 3: 2D constant (or trivial batch dims like (1, k, n))
            return self._rmul_kronecker(rhs_data, const_shape, var_shape, view)

    @staticmethod
    def reshape_constant_data(constant_data: dict, lin_op_shape: tuple) -> dict:
        """Reshape constant data from column format to required shape.

        The input CooTensor is in column format (m*n, 1). We reshape it
        to (m, n) for operations like mul that need the actual shape.

        For parametric data, uses reshape_parametric_constant which handles
        broadcast deduplication based on param_idx.
        """
        result = {}
        for k, v in constant_data.items():
            if isinstance(v, CooTensor):
                new_m = lin_op_shape[0] if len(lin_op_shape) > 0 else 1
                new_n = lin_op_shape[1] if len(lin_op_shape) > 1 else 1
                # Reshape from column (m*n, 1) to matrix (m, n)
                # Use reshape_parametric_constant for proper param_idx handling
                result[k] = reshape_parametric_constant(v, new_m, new_n)
            else:
                result[k] = v.reshape(lin_op_shape, order='F') if hasattr(v, 'reshape') else v
        return result

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """Returns a function to extend and shift a CooTensor."""
        def stack_func(compact, p):
            return CooTensor(
                data=compact.data.copy(),
                row=compact.row + offset,
                col=compact.col.copy(),
                param_idx=compact.param_idx.copy(),
                m=total_rows,
                n=compact.n,
                param_size=compact.param_size
            )
        return stack_func

    def conv(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Discrete convolution.

        Builds a Toeplitz-like matrix from the convolution kernel and multiplies.

        Note: conv currently doesn't support parameters.
        """
        # Compute target shape (2D shape, or row vector for 1D, or (1,1) for 0D)
        data_shape = lin_op.data.shape
        if len(data_shape) == 2:
            target = data_shape
        elif len(data_shape) == 1:
            target = (1, data_shape[0])
        else:  # 0D scalar
            target = (1, 1)
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, target_shape=target)
        assert is_param_free_lhs, "conv doesn't support parametrized kernel"

        # Convert to sparse - may be CooTensor or sparse matrix
        if isinstance(lhs, CooTensor):
            lhs_sparse = lhs.to_stacked_sparse().tocoo()
        elif sp.issparse(lhs):
            lhs_sparse = lhs.tocoo()
        else:
            lhs_arr = np.atleast_2d(lhs)
            lhs_sparse = sp.coo_array(lhs_arr)

        # Need column vector for convolution
        # SciPy returns (1, n) for 1D data and transposes to (n, 1)
        # COO returns (n, 1) already, so check actual shape
        if lhs_sparse.shape[0] == 1 and lhs_sparse.shape[1] > 1:
            # Row vector -> transpose to column
            lhs_sparse = lhs_sparse.T.tocoo()

        rows = lin_op.shape[0]
        cols = lin_op.args[0].shape[0] if len(lin_op.args[0].shape) > 0 else 1

        # Build Toeplitz-like convolution matrix
        nonzeros = lhs_sparse.nnz
        row_idx = (np.tile(lhs_sparse.row, cols) +
                   np.repeat(np.arange(cols), nonzeros)).astype(int)
        col_idx = (np.tile(lhs_sparse.col, cols) +
                   np.repeat(np.arange(cols), nonzeros)).astype(int)
        data = np.tile(lhs_sparse.data, cols)

        conv_matrix = sp.csr_array((data, (row_idx, col_idx)), shape=(rows, cols))
        conv_compact = self._to_coo_tensor(conv_matrix)

        def conv_func(compact, p):
            assert compact.param_size == 1, "conv doesn't support parametrized input"
            return coo_matmul(conv_compact, compact)

        view.accumulate_over_variables(conv_func, is_param_free_function=True)
        return view

    def _kron_impl(self, lin_op, view: CooTensorView, const_on_left: bool) -> CooTensorView:
        """
        Unified Kronecker product implementation.

        If const_on_left=True: computes kron(constant, variable)
        If const_on_left=False: computes kron(variable, constant)

        Reorders rows to match CVXPY's column-major ordering.
        Note: kron currently doesn't support parameters.
        """
        const_data, is_param_free = self.get_constant_data(
            lin_op.data, view, target_shape=None
        )
        assert is_param_free, "kron doesn't support parametrized operands"

        # Convert constant to sparse
        if isinstance(const_data, CooTensor):
            const_sparse = const_data.to_stacked_sparse()
        elif sp.issparse(const_data):
            const_sparse = const_data
        else:
            const_sparse = sp.csr_array(np.asarray(const_data))

        var_shape = lin_op.args[0].shape
        const_shape = lin_op.data.shape

        if const_on_left:
            row_idx = self._get_kron_row_indices(const_shape, var_shape)
        else:
            row_idx = self._get_kron_row_indices(var_shape, const_shape)

        def kron_func(compact, p):
            assert compact.param_size == 1, "kron doesn't support parametrized input"
            x_sparse = compact.to_stacked_sparse()
            if const_on_left:
                kron_res = sp.kron(const_sparse, x_sparse).tocsr()
            else:
                kron_res = sp.kron(x_sparse, const_sparse).tocsr()
            kron_res = kron_res[row_idx, :]
            return CooTensor.from_stacked_sparse(kron_res, param_size=1)

        view.accumulate_over_variables(kron_func, is_param_free_function=True)
        return view

    def kron_r(self, lin_op, view: CooTensorView) -> CooTensorView:
        """Kronecker product kron(a, x) - constant on left."""
        return self._kron_impl(lin_op, view, const_on_left=True)

    def kron_l(self, lin_op, view: CooTensorView) -> CooTensorView:
        """Kronecker product kron(x, a) - constant on right."""
        return self._kron_impl(lin_op, view, const_on_left=False)

    def trace(self, lin_op, view: CooTensorView) -> CooTensorView:
        """
        Compute trace - sum of diagonal elements.

        Uses select_rows to get diagonal + sum_entries.
        """
        # Get shape from argument
        arg_shape = lin_op.args[0].shape
        rows = arg_shape[0]
        # Extract diagonal: indices 0, n+1, 2*(n+1), etc for main diagonal
        diag_indices = np.arange(rows) * (rows + 1)
        view.select_rows(diag_indices.astype(int))
        # Sum entries
        return self.sum_entries(lin_op, view)
