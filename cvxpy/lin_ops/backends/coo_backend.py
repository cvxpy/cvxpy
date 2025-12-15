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

COO (Coordinate) Backend - 3D sparse tensor backend for large parameter problems.

Instead of storing parameter slices as one stacked matrix of shape (param_size * m, n),
this stores data in a compact 3D COO format: (data, row, col, param_idx).

This achieves O(nnz) complexity instead of O(param_size * m) for most operations.

Key differences from SciPyTensorView:
- No stacking: parameter slices are kept separate conceptually
- Operations are O(nnz) instead of O(rows)
- get_tensor_representation() is trivial (data already in right format)

Classes:
- CoordsTensor: 3D sparse COO tensor storage
- CoordsTensorView: TensorView using CoordsTensor
- COOCanonBackend: Backend implementation
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
class CoordsTensor:
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
    def empty(cls, m: int, n: int, param_size: int) -> CoordsTensor:
        """Create an empty CoordsTensor with given dimensions."""
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
    def from_stacked_sparse(cls, matrix: sp.spmatrix, param_size: int) -> CoordsTensor:
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

    def to_tensor_representation(self, row_offset: int, col_offset: int,
                                  param_col_offset: int) -> TensorRepresentation:
        """Convert to TensorRepresentation format."""
        # This is now trivial - just add offsets
        return TensorRepresentation(
            data=self.data,
            row=self.row + row_offset,
            col=self.col + col_offset,
            parameter_offset=self.param_idx + param_col_offset,
            # Shape reflects the final matrix dimensions after applying offsets
            shape=(self.m + row_offset, self.n + col_offset)
        )

    def select_rows(self, rows: np.ndarray) -> CoordsTensor:
        """
        Select and reorder rows from each parameter slice.

        Semantics match SciPy: new_tensor[i, :] = old_tensor[rows[i], :]
        This means `rows` contains the SOURCE row index for each DESTINATION row.

        For example, rows=[2, 0, 1] means:
          - new row 0 <- old row 2
          - new row 1 <- old row 0
          - new row 2 <- old row 1
        """
        new_m = len(rows)

        if self.nnz == 0:
            return CoordsTensor.empty(new_m, self.n, self.param_size)

        # Fast path: no duplicate rows (common case for transpose, reshape, etc.)
        if len(rows) == len(np.unique(rows)):
            return self._select_rows_no_duplicates(rows, new_m)

        # General path: handles duplicate rows (needed for broadcasting, etc.)
        return self._select_rows_with_duplicates(rows, new_m)

    def _select_rows_no_duplicates(self, rows: np.ndarray, new_m: int) -> CoordsTensor:
        """Fast path for select_rows when there are no duplicate rows."""
        # Build reverse mapping: old_row -> new_row (or -1 if not selected)
        row_map = np.full(self.m, -1, dtype=np.int64)
        row_map[rows] = np.arange(new_m, dtype=np.int64)

        # Map tensor rows to new positions
        new_rows = row_map[self.row]

        # Keep only entries whose row is selected
        mask = new_rows >= 0

        return CoordsTensor(
            data=self.data[mask],
            row=new_rows[mask],
            col=self.col[mask],
            param_idx=self.param_idx[mask],
            m=new_m,
            n=self.n,
            param_size=self.param_size
        )

    def _select_rows_with_duplicates(self, rows: np.ndarray, new_m: int) -> CoordsTensor:
        """General path for select_rows that handles duplicate rows.

        Uses binary search (searchsorted) to efficiently find which tensor
        entries need to be replicated. For each unique tensor row, we find all
        destination positions it maps to in O(log n) time via binary search,
        then use np.repeat to duplicate entries accordingly.
        """
        # Sort the rows array to enable efficient range queries
        # rows[new_pos] = old_pos, so we sort to find all new_pos for each old_pos
        row_sort_perm = np.argsort(rows)
        rows_sorted = rows[row_sort_perm]

        # Find unique tensor rows and how many destinations each maps to
        unique_tensor_rows, inverse_idx = np.unique(self.row, return_inverse=True)

        # For each unique tensor row, find the range in sorted rows via searchsorted
        left = np.searchsorted(rows_sorted, unique_tensor_rows, side='left')
        right = np.searchsorted(rows_sorted, unique_tensor_rows, side='right')
        counts = right - left  # How many new rows each unique tensor row maps to

        # Per-entry counts: how many outputs each tensor entry will produce
        entry_counts = counts[inverse_idx]
        total_nnz = entry_counts.sum()

        if total_nnz == 0:
            return CoordsTensor.empty(new_m, self.n, self.param_size)

        # Expand entries according to counts using np.repeat
        out_data = np.repeat(self.data, entry_counts)
        out_col = np.repeat(self.col, entry_counts)
        out_param = np.repeat(self.param_idx, entry_counts)

        # Build output row indices:
        # For each repeated entry, we need to gather from row_sort_perm at the
        # appropriate position within its range [left, right)

        # Get the left boundary for each entry
        entry_lefts = np.repeat(left[inverse_idx], entry_counts)

        # Build position-within-range using cumsum trick
        # First, compute the starting offset for each entry's outputs
        entry_offsets = np.zeros(len(self.data) + 1, dtype=np.int64)
        entry_offsets[1:] = np.cumsum(entry_counts)

        # Position within each entry's range: [0, 1, 2, ...] repeating per entry
        positions = np.arange(total_nnz, dtype=np.int64) - np.repeat(
            entry_offsets[:-1], entry_counts
        )

        # Gather indices into row_sort_perm
        gather_idx = entry_lefts + positions

        # The output rows are the positions in the original rows array
        out_row = row_sort_perm[gather_idx]

        return CoordsTensor(
            data=out_data,
            row=out_row,
            col=out_col,
            param_idx=out_param,
            m=new_m,
            n=self.n,
            param_size=self.param_size
        )

    def scale(self, factor: float) -> CoordsTensor:
        """Scale all values by a constant."""
        return CoordsTensor(
            data=self.data * factor,
            row=self.row,
            col=self.col,
            param_idx=self.param_idx,
            m=self.m,
            n=self.n,
            param_size=self.param_size
        )

    def negate(self) -> CoordsTensor:
        """Negate all values."""
        return self.scale(-1.0)

    def _transpose_helper(self) -> CoordsTensor:
        """2D matrix transpose (swap rows and cols).

        Internal helper for matrix operations like rmul. Not for ND linop transpose.
        """
        return CoordsTensor(
            data=self.data,
            row=self.col,
            col=self.row,
            param_idx=self.param_idx,
            m=self.n,
            n=self.m,
            param_size=self.param_size
        )

    def __add__(self, other: CoordsTensor) -> CoordsTensor:
        """Add two CoordsTensors (concatenate entries)."""
        # Allow different n (column counts) - take max
        # This happens in vstack when combining expressions with different variable counts
        if self.m != other.m:
            raise ValueError(f"Row count mismatch: {self.m} vs {other.m}")
        if self.param_size != other.param_size:
            raise ValueError(f"Param size mismatch: {self.param_size} vs {other.param_size}")

        return CoordsTensor(
            data=np.concatenate([self.data, other.data]),
            row=np.concatenate([self.row, other.row]),
            col=np.concatenate([self.col, other.col]),
            param_idx=np.concatenate([self.param_idx, other.param_idx]),
            m=self.m,
            n=max(self.n, other.n),  # Take max column count
            param_size=self.param_size
        )


def _coo_kron_eye_r(tensor: CoordsTensor, reps: int) -> CoordsTensor:
    """
    Apply Kronecker product kron(I_reps, A) to a CoordsTensor.

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

    return CoordsTensor(
        data=new_data,
        row=new_row,
        col=new_col,
        param_idx=new_param_idx,
        m=m * reps,
        n=k * reps,
        param_size=tensor.param_size
    )


def coo_matmul(lhs: CoordsTensor, rhs: CoordsTensor) -> CoordsTensor:
    """
    Matrix multiplication of two CoordsTensors.

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
            return CoordsTensor.empty(lhs.m, rhs.n, lhs.param_size)

        # Pre-allocate output
        out_data = np.empty(total_nnz, dtype=np.float64)
        out_row = np.empty(total_nnz, dtype=np.int64)
        out_col = np.empty(total_nnz, dtype=np.int64)
        out_param = np.empty(total_nnz, dtype=np.int64)

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

        return CoordsTensor(
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
            return CoordsTensor.empty(lhs.m, rhs.n, rhs.param_size)

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

        return CoordsTensor(
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

        return CoordsTensor(
            data=result.data.copy(),
            row=result.row.astype(np.int64),
            col=result.col.astype(np.int64),
            param_idx=np.zeros(len(result.data), dtype=np.int64),
            m=lhs.m,
            n=rhs.n,
            param_size=1
        )

    else:
        # Both parametrized - need to match param_idx
        # This is rare in DPP, fall back to per-slice computation
        if lhs.param_size != rhs.param_size:
            raise ValueError("Mismatched param_size in coo_matmul")

        results = []
        for p in range(lhs.param_size):
            # Extract slice p from both
            lhs_mask = lhs.param_idx == p
            rhs_mask = rhs.param_idx == p

            lhs_slice = sp.csr_array(
                (lhs.data[lhs_mask], (lhs.row[lhs_mask], lhs.col[lhs_mask])),
                shape=(lhs.m, lhs.n)
            )
            rhs_slice = sp.csc_array(
                (rhs.data[rhs_mask], (rhs.row[rhs_mask], rhs.col[rhs_mask])),
                shape=(rhs.m, rhs.n)
            )
            result_slice = (lhs_slice @ rhs_slice).tocoo()

            if result_slice.nnz > 0:
                results.append((
                    result_slice.data,
                    result_slice.row,
                    result_slice.col,
                    np.full(result_slice.nnz, p, dtype=np.int64)
                ))

        if not results:
            return CoordsTensor.empty(lhs.m, rhs.n, lhs.param_size)

        all_data = np.concatenate([r[0] for r in results])
        all_row = np.concatenate([r[1] for r in results]).astype(np.int64)
        all_col = np.concatenate([r[2] for r in results]).astype(np.int64)
        all_param = np.concatenate([r[3] for r in results])

        return CoordsTensor(
            data=all_data,
            row=all_row,
            col=all_col,
            param_idx=all_param,
            m=lhs.m,
            n=rhs.n,
            param_size=lhs.param_size
        )


def coo_mul_elem(lhs: CoordsTensor, rhs: CoordsTensor) -> CoordsTensor:
    """
    Element-wise multiplication of two CoordsTensors.

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
                return CoordsTensor.empty(lhs.m, lhs.n, lhs.param_size)
            return CoordsTensor(
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
                return CoordsTensor.empty(lhs.m, rhs.n, lhs.param_size)

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

            return CoordsTensor(
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

            # For each lhs entry, get the corresponding rhs value at same row
            lhs_rows = lhs.row

            # Check if each lhs row has a matching rhs entry
            # rhs is column vector, so each row has at most one entry
            out_data = []
            out_row = []
            out_col = []
            out_param = []

            for i in range(len(lhs.data)):
                row = lhs_rows[i]
                start, end = rhs_indptr[row], rhs_indptr[row + 1]
                if start < end:
                    # There's an rhs entry at this row
                    rhs_val = rhs_data_sorted[start]
                    out_data.append(lhs.data[i] * rhs_val)
                    out_row.append(lhs.row[i])
                    out_col.append(lhs.col[i])
                    out_param.append(lhs.param_idx[i])
                # If no rhs entry at this row, result is 0 (skip)

            if len(out_data) == 0:
                return CoordsTensor.empty(lhs.m, lhs.n, lhs.param_size)

            return CoordsTensor(
                data=np.array(out_data),
                row=np.array(out_row, dtype=np.int64),
                col=np.array(out_col, dtype=np.int64),
                param_idx=np.array(out_param, dtype=np.int64),
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
        return CoordsTensor(
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
        return CoordsTensor(
            data=result.data.copy(),
            row=result.row.copy(),
            col=result.col.copy(),
            param_idx=np.zeros(len(result.data), dtype=np.int64),
            m=lhs.m,
            n=lhs.n,
            param_size=1
        )

    else:
        # Both parametrized - element-wise multiply matching slices
        # Vectorized: sorted merge with searchsorted
        if lhs.param_size != rhs.param_size:
            raise ValueError("Mismatched param_size in coo_mul_elem")

        # Compute linear indices: param_idx * (m * n) + row * n + col
        slice_size = lhs.m * lhs.n
        lhs_linear = lhs.param_idx * slice_size + lhs.row * lhs.n + lhs.col
        rhs_linear = rhs.param_idx * slice_size + rhs.row * rhs.n + rhs.col

        # Sort rhs for binary search
        rhs_sort = np.argsort(rhs_linear)
        rhs_sorted = rhs_linear[rhs_sort]

        # Find matching indices in rhs for each lhs entry
        match_pos = np.searchsorted(rhs_sorted, lhs_linear)

        # Check which matches are valid (within bounds and actually equal)
        match_pos_clipped = np.minimum(match_pos, len(rhs_sorted) - 1)
        valid = (match_pos < len(rhs_sorted)) & (rhs_sorted[match_pos_clipped] == lhs_linear)

        if not valid.any():
            return CoordsTensor.empty(lhs.m, lhs.n, lhs.param_size)

        # Get matching rhs indices in original order
        rhs_match_idx = rhs_sort[match_pos[valid]]

        return CoordsTensor(
            data=lhs.data[valid] * rhs.data[rhs_match_idx],
            row=lhs.row[valid],
            col=lhs.col[valid],
            param_idx=lhs.param_idx[valid],
            m=lhs.m,
            n=lhs.n,
            param_size=lhs.param_size
        )


def coo_reshape(tensor: CoordsTensor, new_m: int, new_n: int) -> CoordsTensor:
    """
    Reshape the tensor (Fortran order, column-major).

    For each entry at (param_idx, row, col), compute linear index = col * m + row,
    then new_row = linear_idx % new_m, new_col = linear_idx // new_m.
    """
    # Compute linear index in column-major order
    linear_idx = tensor.col * tensor.m + tensor.row

    # Compute new row and col
    new_row = linear_idx % new_m
    new_col = linear_idx // new_m

    return CoordsTensor(
        data=tensor.data.copy(),
        row=new_row.astype(np.int64),
        col=new_col.astype(np.int64),
        param_idx=tensor.param_idx.copy(),
        m=new_m,
        n=new_n,
        param_size=tensor.param_size
    )


class CoordsTensorView(DictTensorView):
    """
    TensorView using CoordsTensor storage for O(nnz) operations.

    Unlike SciPyTensorView which stores stacked sparse matrices of shape
    (param_size * m, n), CoordsTensorView stores CoordsTensor objects that
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

        This is trivial for CoordsTensor - just add offsets.
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

        func signature: func(compact: CoordsTensor, p: int) -> CoordsTensor
        """
        self.tensor = {var_id: {k: func(v, self.param_to_size[k])
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> 'CoordsTensorView':
        """Create new CoordsTensorView with same shape information."""
        return CoordsTensorView(
            variable_ids, tensor, is_parameter_free,
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, CoordsTensor]) \
            -> dict[int, CoordsTensor]:
        """Apply 'func' to each parameter slice."""
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: CoordsTensor, b: CoordsTensor) -> CoordsTensor:
        """Add two CoordsTensors."""
        return a + b

    @staticmethod
    def tensor_type():
        """The tensor type for CoordsTensorView."""
        return CoordsTensor


class COOCanonBackend(PythonCanonBackend):
    """
    Canon backend using CoordsTensorView for O(nnz) operations.

    This backend stores tensors in compact COO format with separate
    parameter indices, avoiding the creation of huge stacked matrices.
    """

    def get_empty_view(self) -> CoordsTensorView:
        """Return an empty CoordsTensorView."""
        return CoordsTensorView.get_empty_view(
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def get_variable_tensor(self, shape: tuple, var_id: int) -> dict:
        """
        Create tensor for a variable.

        Returns {var_id: {Constant.ID: tensor}} where tensor is identity-like.
        """
        size = int(np.prod(shape))
        compact = CoordsTensor(
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
            data = data.toarray()
        flat = np.asarray(data).flatten(order='F')
        size = len(flat)

        # Find non-zero entries
        nz_mask = flat != 0
        nz_idx = np.where(nz_mask)[0]
        nz_data = flat[nz_mask]

        compact = CoordsTensor(
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
        compact = CoordsTensor(
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

    def neg(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Negate all values."""
        def func(compact, p):
            return compact.negate()
        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    def sum_entries(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
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
                return CoordsTensor(
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
            new_m = int(np.prod(shape) // np.prod([shape[a] for a in
                        (axis if isinstance(axis, tuple) else (axis,))]))

            def func(compact, p):
                return CoordsTensor(
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

    def reshape(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Reshape tensor (column-major order)."""
        new_shape = lin_op.shape
        new_m = int(np.prod(new_shape))

        def func(compact, p):
            return coo_reshape(compact, new_m, compact.n)

        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    # transpose: use base class implementation (via select_rows)

    def mul(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """
        Matrix multiplication - the key optimized operation.

        For parametrized A @ x, this is O(nnz) instead of O(param_size * m).
        """
        lhs = lin_op.data

        # Get constant data for lhs
        lhs_data, is_lhs_parametric = self._get_lhs_data(lhs, view)

        if is_lhs_parametric:
            # Parametrized lhs @ variable rhs - the expensive case
            # Need to apply Kronecker expansion for ND variables
            lhs_shape = lhs.shape
            # For 1D lhs in matmul, treat as row vector (1, size) to match numpy behavior
            lhs_shape_2d = lhs_shape if len(lhs_shape) == 2 else (1, int(np.prod(lhs_shape)))
            lhs_k = lhs_shape_2d[-1]  # Inner dimension
            reps = view.rows // lhs_k

            # Apply Kronecker expansion if needed
            if reps > 1:
                expanded_lhs = {
                    param_id: _coo_kron_eye_r(tensor, reps)
                    for param_id, tensor in lhs_data.items()
                }
            else:
                expanded_lhs = lhs_data

            def parametrized_mul(rhs_compact):
                # lhs_data is a dict {param_id: CoordsTensor}
                result = {}
                for param_id, lhs_compact in expanded_lhs.items():
                    result[param_id] = coo_matmul(lhs_compact, rhs_compact)
                return result

            # Apply to each variable tensor in-place
            for var_id, var_tensor in view.tensor.items():
                # var_tensor is {Constant.ID: CoordsTensor}
                const_compact = var_tensor[Constant.ID.value]
                view.tensor[var_id] = parametrized_mul(const_compact)

            view.is_parameter_free = False
            return view

        else:
            # Constant lhs @ rhs - need to expand lhs with Kronecker product
            # For A @ X where A is (m,k) and X is variable:
            # We need kron(I_reps, A) where reps = X_rows / k
            lhs_shape = lin_op.data.shape
            lhs_shape_2d = lhs_shape if len(lhs_shape) == 2 else (1, lhs_shape[0])
            lhs_k = lhs_shape_2d[-1]  # Inner dimension of A

            # Convert to sparse and apply kron expansion
            # get_constant_data may return CoordsTensor, sparse, or dense array
            if isinstance(lhs_data, CoordsTensor):
                lhs_sparse = lhs_data.to_stacked_sparse()
            elif sp.issparse(lhs_data):
                lhs_sparse = lhs_data
            else:
                lhs_sparse = sp.csr_array(np.atleast_2d(lhs_data))

            reps = view.rows // lhs_k
            if reps > 1:
                stacked_lhs = sp.kron(sp.eye_array(reps, format="csr"), lhs_sparse)
            else:
                stacked_lhs = lhs_sparse

            # Convert stacked lhs to CoordsTensor
            stacked_compact = self._to_coo_tensor(stacked_lhs)

            def constant_mul(compact, p):
                return coo_matmul(stacked_compact, compact)

            view.accumulate_over_variables(constant_mul, is_param_free_function=True)
            return view

    def _get_lhs_data(self, lhs, view):
        """Get lhs data, detecting if it's parametric."""
        if hasattr(lhs, 'type') and lhs.type == 'param':
            # Parametric lhs
            param_id = lhs.data
            param_size = self.param_to_size[param_id]
            size = int(np.prod(lhs.shape))
            # For 1D lhs in matmul, treat as row vector (1, size) to match numpy behavior
            m, k = lhs.shape if len(lhs.shape) == 2 else (1, size)

            # Create CoordsTensor for parameter matrix
            # Parameters are stored in column-major (Fortran) order:
            # param_idx=0 -> A[0,0], param_idx=1 -> A[1,0], ..., param_idx=m -> A[0,1]
            compact = CoordsTensor(
                data=np.ones(param_size, dtype=np.float64),
                row=np.tile(np.arange(m), k),  # [0,1,2,0,1,2,...] for column-major
                col=np.repeat(np.arange(k), m),  # [0,0,0,1,1,1,...] for column-major
                param_idx=np.arange(param_size, dtype=np.int64),
                m=m,
                n=k,
                param_size=param_size
            )
            return {param_id: compact}, True
        else:
            # Constant lhs - use get_constant_data to handle complex expressions
            lhs_data, is_param_free = self.get_constant_data(lhs, view, column=False)
            return lhs_data, not is_param_free  # return (data, is_parametric)

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

    def div(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """
        Division by constant: x / d.

        Note: div currently doesn't support parameters in divisor.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=True)
        assert is_param_free_lhs, "div doesn't support parametrized divisor"

        # Get reciprocal values
        lhs_compact = self._to_coo_tensor(lhs)
        # Check for zero divisors
        if np.any(lhs_compact.data == 0):
            raise ValueError("Division by zero encountered in divisor")
        # Invert the data
        recip_data = np.reciprocal(lhs_compact.data, dtype=float)
        lhs_recip = CoordsTensor(
            data=recip_data,
            row=lhs_compact.row,
            col=lhs_compact.col,
            param_idx=lhs_compact.param_idx,
            m=lhs_compact.m,
            n=lhs_compact.n,
            param_size=lhs_compact.param_size
        )

        def div_func(compact, p):
            return coo_mul_elem(compact, lhs_recip)

        view.accumulate_over_variables(div_func, is_param_free_function=True)
        return view

    def mul_elem(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """
        Element-wise multiplication: x * d.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=True)

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
    def promote(lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Promote scalar by repeating."""
        num_entries = int(np.prod(lin_op.shape))
        rows = np.zeros(num_entries, dtype=np.int64)
        view.select_rows(rows)
        return view

    @staticmethod
    def broadcast_to(lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Broadcast to shape."""
        broadcast_shape = lin_op.shape
        original_shape = lin_op.args[0].shape
        rows = np.arange(np.prod(original_shape, dtype=int)).reshape(original_shape, order='F')
        rows = np.broadcast_to(rows, broadcast_shape).flatten(order="F")
        view.select_rows(rows.astype(np.int64))
        return view

    # index: use base class implementation (via select_rows)

    def diag_vec(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
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

            return CoordsTensor(
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
        """Convert sparse matrix to CoordsTensor."""
        if isinstance(matrix, dict):
            # Already a dict of matrices
            raise ValueError("Expected single matrix, got dict")

        coo = matrix.tocoo()
        if param_id is not None:
            p = self.param_to_size[param_id]
            m = coo.shape[0] // p
            param_idx, row = np.divmod(coo.row, m)
        else:
            p = 1
            m = coo.shape[0]
            row = coo.row
            param_idx = np.zeros(len(coo.data), dtype=np.int64)

        return CoordsTensor(
            data=coo.data.copy(),
            row=row.astype(np.int64),
            col=coo.col.astype(np.int64),
            param_idx=param_idx.astype(np.int64),
            m=m,
            n=coo.shape[1],
            param_size=p
        )

    # vstack, hstack, concatenate: use base class implementations

    def rmul(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """
        Right multiplication: x @ B.

        For X @ B where X is (m, k) variable and B is (k, n) constant:
        vec(X @ B) = (B.T ⊗ I_m) @ vec(X)

        Supports both constant and parametrized B.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=False)

        # Get dimensions
        arg_shape = lin_op.args[0].shape
        arg_cols = arg_shape[0] if len(arg_shape) == 1 else arg_shape[1]

        if is_param_free_lhs:
            # Constant B case
            # Convert to sparse - may be CoordsTensor or sparse matrix
            if isinstance(lhs, CoordsTensor):
                lhs_sparse = lhs.to_stacked_sparse()
            elif sp.issparse(lhs):
                lhs_sparse = lhs
            else:
                lhs_sparse = sp.csr_array(np.atleast_2d(lhs))

            # Handle 1D case - may need transpose
            if len(lin_op.data.shape) == 1 and arg_cols != lhs_sparse.shape[0]:
                lhs_sparse = lhs_sparse.T

            # Compute kron(B.T, I_reps) where reps = X_rows / B_rows
            reps = view.rows // lhs_sparse.shape[0]
            if reps > 1:
                stacked_lhs = sp.kron(lhs_sparse.T, sp.eye_array(reps, format="csr"))
            else:
                stacked_lhs = lhs_sparse.T

            # Convert to CoordsTensor
            stacked_compact = self._to_coo_tensor(stacked_lhs)

            def rmul_func(compact, p):
                return coo_matmul(stacked_compact, compact)

            view.accumulate_over_variables(rmul_func, is_param_free_function=True)
            return view
        else:
            # Parametrized B case
            # lhs is dict {param_id: CoordsTensor}

            # Get representative param slice to determine dimensions
            param_id, first_compact = next(iter(lhs.items()))
            lhs_rows = first_compact.m

            # Handle 1D case - may need transpose
            if len(lin_op.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = {k: v._transpose_helper() for k, v in lhs.items()}
                param_id, first_compact = next(iter(lhs.items()))
                lhs_rows = first_compact.m

            reps = view.rows // lhs_rows

            # Transpose each param slice (B.T)
            lhs_transposed = {k: v._transpose_helper() for k, v in lhs.items()}

            # Apply kron expansion if needed
            if reps > 1:
                # kron(B.T, I_reps) for each param slice
                stacked_lhs = {}
                for k, v in lhs_transposed.items():
                    v_sparse = v.to_stacked_sparse()
                    kron_result = sp.kron(v_sparse, sp.eye_array(reps, format="csr"))
                    stacked_lhs[k] = self._to_coo_tensor(kron_result, param_id=k)
            else:
                stacked_lhs = {k: self._to_coo_tensor(v.to_stacked_sparse(), param_id=k)
                               for k, v in lhs_transposed.items()}

            def parametrized_rmul(rhs_compact):
                # Multiply each param slice of stacked_lhs with the constant rhs
                return {k: coo_matmul(v, rhs_compact) for k, v in stacked_lhs.items()}

            # Apply to each variable tensor in-place
            for var_id, var_tensor in view.tensor.items():
                const_compact = var_tensor[Constant.ID.value]
                view.tensor[var_id] = parametrized_rmul(const_compact)

            view.is_parameter_free = False
            return view

    @staticmethod
    def reshape_constant_data(constant_data: dict, lin_op_shape: tuple) -> dict:
        """Reshape constant data from column format to required shape.

        The input CoordsTensor is in column format (m*n, 1). We reshape it
        to (m, n) for operations like mul that need the actual shape.
        """
        result = {}
        for k, v in constant_data.items():
            if isinstance(v, CoordsTensor):
                new_m = lin_op_shape[0] if len(lin_op_shape) > 0 else 1
                new_n = lin_op_shape[1] if len(lin_op_shape) > 1 else 1
                # Reshape from column (m*n, 1) to matrix (m, n)
                result[k] = coo_reshape(v, new_m, new_n)
            else:
                result[k] = v.reshape(lin_op_shape, order='F') if hasattr(v, 'reshape') else v
        return result

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """Returns a function to extend and shift a CoordsTensor."""
        def stack_func(compact, p):
            return CoordsTensor(
                data=compact.data.copy(),
                row=compact.row + offset,
                col=compact.col.copy(),
                param_idx=compact.param_idx.copy(),
                m=total_rows,
                n=compact.n,
                param_size=compact.param_size
            )
        return stack_func

    def conv(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """
        Discrete convolution.

        Builds a Toeplitz-like matrix from the convolution kernel and multiplies.

        Note: conv currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=False)
        assert is_param_free_lhs, "conv doesn't support parametrized kernel"

        # Convert to sparse - may be CoordsTensor or sparse matrix
        if isinstance(lhs, CoordsTensor):
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

    def _kron_impl(self, lin_op, view: CoordsTensorView, const_on_left: bool) -> CoordsTensorView:
        """
        Unified Kronecker product implementation.

        If const_on_left=True: computes kron(constant, variable)
        If const_on_left=False: computes kron(variable, constant)

        Reorders rows to match CVXPY's column-major ordering.
        Note: kron currently doesn't support parameters.
        """
        const_data, is_param_free = self.get_constant_data(lin_op.data, view, column=True)
        assert is_param_free, "kron doesn't support parametrized operands"

        # Convert constant to sparse
        if isinstance(const_data, CoordsTensor):
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
            return CoordsTensor.from_stacked_sparse(kron_res, param_size=1)

        view.accumulate_over_variables(kron_func, is_param_free_function=True)
        return view

    def kron_r(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Kronecker product kron(a, x) - constant on left."""
        return self._kron_impl(lin_op, view, const_on_left=True)

    def kron_l(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
        """Kronecker product kron(x, a) - constant on right."""
        return self._kron_impl(lin_op, view, const_on_left=False)

    def trace(self, lin_op, view: CoordsTensorView) -> CoordsTensorView:
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
