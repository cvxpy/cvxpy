"""
Copyright 2022, the CVXPY authors.

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
- CompactTensor: 3D sparse COO tensor storage
- COOTensorView: TensorView using CompactTensor
- COOCanonBackend: Backend implementation

Note: Previously named "Lazy" backend - renamed to COO to better describe the
storage format (not lazy evaluation, but COO-based 3D tensor representation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops.backends import (
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
)


@dataclass
class CompactTensor:
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

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 2

    @classmethod
    def from_stacked_sparse(cls, matrix: sp.spmatrix, param_size: int) -> CompactTensor:
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
            shape=(self.m + row_offset, self.n + col_offset)  # Approximate
        )

    def select_rows(self, rows: np.ndarray) -> CompactTensor:
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

        # Build reverse mapping: old_row -> list of new_row positions
        # Since rows[new_pos] = old_pos, we need old_pos -> new_pos
        # Multiple new rows can come from the same old row (broadcasting)
        old_to_new = {}
        for new_pos, old_pos in enumerate(rows):
            if old_pos not in old_to_new:
                old_to_new[old_pos] = []
            old_to_new[old_pos].append(new_pos)

        # For each entry in the tensor, map its old row to new row(s)
        new_data = []
        new_rows = []
        new_cols = []
        new_params = []

        for i in range(self.nnz):
            old_row = self.row[i]
            if old_row in old_to_new:
                for new_pos in old_to_new[old_row]:
                    new_data.append(self.data[i])
                    new_rows.append(new_pos)
                    new_cols.append(self.col[i])
                    new_params.append(self.param_idx[i])

        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return CompactTensor(
            data=np.array(new_data, dtype=np.float64) if new_data else empty_f,
            row=np.array(new_rows, dtype=np.int64) if new_rows else empty_i,
            col=np.array(new_cols, dtype=np.int64) if new_cols else empty_i,
            param_idx=np.array(new_params, dtype=np.int64) if new_params else empty_i,
            m=new_m,
            n=self.n,
            param_size=self.param_size
        )

    def scale(self, factor: float) -> CompactTensor:
        """Scale all values by a constant."""
        return CompactTensor(
            data=self.data * factor,
            row=self.row.copy(),
            col=self.col.copy(),
            param_idx=self.param_idx.copy(),
            m=self.m,
            n=self.n,
            param_size=self.param_size
        )

    def negate(self) -> CompactTensor:
        """Negate all values."""
        return self.scale(-1.0)

    def __add__(self, other: CompactTensor) -> CompactTensor:
        """Add two CompactTensors (concatenate entries)."""
        # Allow different n (column counts) - take max
        # This happens in vstack when combining expressions with different variable counts
        assert self.m == other.m, f"Row count mismatch: {self.m} vs {other.m}"
        assert self.param_size == other.param_size, \
            f"Param size mismatch: {self.param_size} vs {other.param_size}"

        return CompactTensor(
            data=np.concatenate([self.data, other.data]),
            row=np.concatenate([self.row, other.row]),
            col=np.concatenate([self.col, other.col]),
            param_idx=np.concatenate([self.param_idx, other.param_idx]),
            m=self.m,
            n=max(self.n, other.n),  # Take max column count
            param_size=self.param_size
        )


def compact_matmul(lhs: CompactTensor, rhs: CompactTensor) -> CompactTensor:
    """
    Matrix multiplication of two CompactTensors.

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

        # Build CSR for rhs for fast row access
        rhs_csr = sp.csr_array(
            (rhs.data, (rhs.row, rhs.col)),
            shape=(rhs.m, rhs.n)
        )
        rhs_indptr = rhs_csr.indptr
        rhs_indices = rhs_csr.indices
        rhs_data = rhs_csr.data

        # For each lhs entry, find how many rhs entries it will generate
        lhs_cols = lhs.col  # These index into rhs rows
        nnz_per_lhs = np.diff(rhs_indptr)[lhs_cols]  # nnz in each rhs row
        total_nnz = nnz_per_lhs.sum()

        if total_nnz == 0:
            return CompactTensor(
                data=np.array([], dtype=np.float64),
                row=np.array([], dtype=np.int64),
                col=np.array([], dtype=np.int64),
                param_idx=np.array([], dtype=np.int64),
                m=lhs.m,
                n=rhs.n,
                param_size=lhs.param_size
            )

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

        return CompactTensor(
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

        # Build CSC for lhs for fast column access
        lhs_csc = sp.csc_array(
            (lhs.data, (lhs.row, lhs.col)),
            shape=(lhs.m, lhs.n)
        )
        lhs_indptr = lhs_csc.indptr
        lhs_indices = lhs_csc.indices  # row indices
        lhs_data = lhs_csc.data

        # For each rhs entry, find how many lhs entries it will generate
        rhs_rows = rhs.row  # These index into lhs columns
        nnz_per_rhs = np.diff(lhs_indptr)[rhs_rows]  # nnz in each lhs column
        total_nnz = nnz_per_rhs.sum()

        if total_nnz == 0:
            return CompactTensor(
                data=np.array([], dtype=np.float64),
                row=np.array([], dtype=np.int64),
                col=np.array([], dtype=np.int64),
                param_idx=np.array([], dtype=np.int64),
                m=lhs.m,
                n=rhs.n,
                param_size=rhs.param_size
            )

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

        return CompactTensor(
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

        return CompactTensor(
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
        assert lhs.param_size == rhs.param_size, \
            "Mismatched param_size in compact_matmul"

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
            return CompactTensor(
                data=np.array([], dtype=np.float64),
                row=np.array([], dtype=np.int64),
                col=np.array([], dtype=np.int64),
                param_idx=np.array([], dtype=np.int64),
                m=lhs.m,
                n=rhs.n,
                param_size=lhs.param_size
            )

        all_data = np.concatenate([r[0] for r in results])
        all_row = np.concatenate([r[1] for r in results]).astype(np.int64)
        all_col = np.concatenate([r[2] for r in results]).astype(np.int64)
        all_param = np.concatenate([r[3] for r in results])

        return CompactTensor(
            data=all_data,
            row=all_row,
            col=all_col,
            param_idx=all_param,
            m=lhs.m,
            n=rhs.n,
            param_size=lhs.param_size
        )


def compact_mul_elem(lhs: CompactTensor, rhs: CompactTensor) -> CompactTensor:
    """
    Element-wise multiplication of two CompactTensors.

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
                return CompactTensor(
                    data=np.array([], dtype=np.float64),
                    row=np.array([], dtype=np.int64),
                    col=np.array([], dtype=np.int64),
                    param_idx=np.array([], dtype=np.int64),
                    m=lhs.m,
                    n=lhs.n,
                    param_size=lhs.param_size
                )
            return CompactTensor(
                data=lhs.data * scalar_val,
                row=lhs.row.copy(),
                col=lhs.col.copy(),
                param_idx=lhs.param_idx.copy(),
                m=lhs.m,
                n=lhs.n,
                param_size=lhs.param_size
            )

        # General case: rhs has same dimensions as lhs slices
        # Build rhs linear mapping for fast lookup
        rhs_linear = rhs.row * rhs.n + rhs.col
        rhs_lookup = np.zeros(rhs.m * rhs.n)
        rhs_lookup[rhs_linear] = rhs.data

        # For each lhs entry, get the corresponding rhs value
        # Handle dimension mismatch by clamping indices
        row_idx = np.minimum(lhs.row, rhs.m - 1)
        col_idx = np.minimum(lhs.col, rhs.n - 1)
        linear_idx = row_idx * rhs.n + col_idx

        # Get rhs values at lhs positions
        rhs_vals = rhs_lookup[linear_idx]

        # Keep only non-zero results
        mask = rhs_vals != 0
        return CompactTensor(
            data=lhs.data[mask] * rhs_vals[mask],
            row=lhs.row[mask].copy(),
            col=lhs.col[mask].copy(),
            param_idx=lhs.param_idx[mask].copy(),
            m=lhs.m,
            n=lhs.n,
            param_size=lhs.param_size
        )

    elif lhs.param_size == 1 and rhs.param_size > 1:
        # Swap and recurse
        return compact_mul_elem(rhs, lhs)

    elif lhs.param_size == 1 and rhs.param_size == 1:
        # Both constant - standard sparse element-wise product
        lhs_csr = lhs.to_stacked_sparse()
        rhs_csr = rhs.to_stacked_sparse()
        result = lhs_csr.multiply(rhs_csr).tocoo()
        return CompactTensor(
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
        assert lhs.param_size == rhs.param_size, \
            "Mismatched param_size in compact_mul_elem"

        # Build lookup for rhs by (param_idx, row, col) -> data
        # Use linear indexing: param_idx * (m * n) + row * n + col
        rhs_linear = (rhs.param_idx * (rhs.m * rhs.n) +
                      rhs.row * rhs.n + rhs.col)
        rhs_lookup = {}
        for i, lin_idx in enumerate(rhs_linear):
            rhs_lookup[lin_idx] = rhs.data[i]

        # For each lhs entry, find matching rhs entry
        lhs_linear = (lhs.param_idx * (lhs.m * lhs.n) +
                      lhs.row * lhs.n + lhs.col)

        out_data = []
        out_row = []
        out_col = []
        out_param = []

        for i, lin_idx in enumerate(lhs_linear):
            if lin_idx in rhs_lookup:
                out_data.append(lhs.data[i] * rhs_lookup[lin_idx])
                out_row.append(lhs.row[i])
                out_col.append(lhs.col[i])
                out_param.append(lhs.param_idx[i])

        return CompactTensor(
            data=np.array(out_data, dtype=np.float64),
            row=np.array(out_row, dtype=np.int64),
            col=np.array(out_col, dtype=np.int64),
            param_idx=np.array(out_param, dtype=np.int64),
            m=lhs.m,
            n=lhs.n,
            param_size=lhs.param_size
        )


def compact_reshape(tensor: CompactTensor, new_m: int, new_n: int) -> CompactTensor:
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

    return CompactTensor(
        data=tensor.data.copy(),
        row=new_row.astype(np.int64),
        col=new_col.astype(np.int64),
        param_idx=tensor.param_idx.copy(),
        m=new_m,
        n=new_n,
        param_size=tensor.param_size
    )


class COOTensorView(DictTensorView):
    """
    TensorView using CompactTensor storage for O(nnz) operations.

    Unlike SciPyTensorView which stores stacked sparse matrices of shape
    (param_size * m, n), COOTensorView stores CompactTensor objects that
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

        This is trivial for CompactTensor - just add offsets.
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

        func signature: func(compact: CompactTensor, p: int) -> CompactTensor
        """
        self.tensor = {var_id: {k: func(v, self.param_to_size[k])
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> 'COOTensorView':
        """Create new COOTensorView with same shape information."""
        return COOTensorView(
            variable_ids, tensor, is_parameter_free,
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, CompactTensor]) \
            -> dict[int, CompactTensor]:
        """Apply 'func' to each parameter slice."""
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: CompactTensor, b: CompactTensor) -> CompactTensor:
        """Add two CompactTensors."""
        return a + b

    @staticmethod
    def tensor_type():
        """The tensor type for COOTensorView."""
        return CompactTensor


class COOCanonBackend(PythonCanonBackend):
    """
    Canon backend using COOTensorView for O(nnz) operations.

    This backend stores tensors in compact COO format with separate
    parameter indices, avoiding the creation of huge stacked matrices.
    """

    def get_empty_view(self) -> COOTensorView:
        """Return an empty COOTensorView."""
        return COOTensorView.get_empty_view(
            self.param_size_plus_one, self.id_to_col,
            self.param_to_size, self.param_to_col, self.var_length
        )

    def get_variable_tensor(self, shape: tuple, var_id: int) -> dict:
        """
        Create tensor for a variable.

        Returns {var_id: {Constant.ID: tensor}} where tensor is identity-like.
        """
        size = int(np.prod(shape))
        compact = CompactTensor(
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

        compact = CompactTensor(
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
        compact = CompactTensor(
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

    def neg(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Negate all values."""
        def func(compact, p):
            return compact.negate()
        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    def sum_entries(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Sum all entries to scalar."""
        def func(compact, p):
            # Sum to scalar: all rows become 0, but columns stay the same
            # (columns represent which variables contribute to the output)
            return CompactTensor(
                data=compact.data.copy(),
                row=np.zeros(compact.nnz, dtype=np.int64),
                col=compact.col.copy(),  # Keep column indices - they're variable indices
                param_idx=compact.param_idx.copy(),
                m=1,
                n=compact.n,
                param_size=compact.param_size
            )
        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    def reshape(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Reshape tensor (column-major order)."""
        new_shape = lin_op.shape
        new_m = int(np.prod(new_shape))

        def func(compact, p):
            return compact_reshape(compact, new_m, compact.n)

        view.accumulate_over_variables(func, is_param_free_function=True)
        return view

    # transpose: use base class implementation (via select_rows)

    def mul(self, lin_op, view: COOTensorView) -> COOTensorView:
        """
        Matrix multiplication - the key optimized operation.

        For parametrized A @ x, this is O(nnz) instead of O(param_size * m).
        """
        lhs = lin_op.data

        # Get constant data for lhs
        lhs_data, is_lhs_parametric = self._get_lhs_data(lhs, view)

        if is_lhs_parametric:
            # Parametrized lhs @ variable rhs - the expensive case
            def parametrized_mul(rhs_compact):
                # lhs_data is a dict {param_id: CompactTensor}
                result = {}
                for param_id, lhs_compact in lhs_data.items():
                    result[param_id] = compact_matmul(lhs_compact, rhs_compact)
                return result

            # Apply to each variable tensor
            new_tensor = {}
            for var_id, var_tensor in view.tensor.items():
                # var_tensor is {Constant.ID: CompactTensor}
                const_compact = var_tensor[Constant.ID.value]
                new_tensor[var_id] = parametrized_mul(const_compact)

            return view.create_new_tensor_view(
                view.variable_ids, new_tensor, is_parameter_free=False
            )

        else:
            # Constant lhs @ rhs - need to expand lhs with Kronecker product
            # For A @ X where A is (m,k) and X is variable:
            # We need kron(I_reps, A) where reps = X_rows / k
            lhs_shape = lin_op.data.shape
            lhs_shape_2d = lhs_shape if len(lhs_shape) == 2 else (1, lhs_shape[0])
            lhs_k = lhs_shape_2d[-1]  # Inner dimension of A

            # Convert to sparse and apply kron expansion
            if sp.issparse(lhs_data):
                lhs_sparse = lhs_data
            else:
                lhs_sparse = sp.csr_array(np.atleast_2d(lhs_data))

            reps = view.rows // lhs_k
            if reps > 1:
                stacked_lhs = sp.kron(sp.eye_array(reps, format="csr"), lhs_sparse)
            else:
                stacked_lhs = lhs_sparse

            # Convert stacked lhs to CompactTensor
            stacked_compact = self._make_compact_from_sparse(stacked_lhs)

            def constant_mul(compact, p):
                return compact_matmul(stacked_compact, compact)

            view.accumulate_over_variables(constant_mul, is_param_free_function=True)
            return view

    def _get_lhs_data(self, lhs, view):
        """Get lhs data, detecting if it's parametric."""
        if hasattr(lhs, 'type') and lhs.type == 'param':
            # Parametric lhs
            param_id = lhs.data
            param_size = self.param_to_size[param_id]
            size = int(np.prod(lhs.shape))
            m, k = lhs.shape if len(lhs.shape) == 2 else (size, 1)

            # Create CompactTensor for parameter matrix
            # Parameters are stored in column-major (Fortran) order:
            # param_idx=0 -> A[0,0], param_idx=1 -> A[1,0], ..., param_idx=m -> A[0,1]
            compact = CompactTensor(
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
            # Constant lhs
            return self.get_constant_data_from_const(lhs), False

    def _make_compact_from_dense(self, data, target_shape):
        """Convert dense/sparse data to CompactTensor.

        Args:
            data: Dense or sparse array
            target_shape: Tuple (m, n) specifying the desired shape
        """
        if sp.issparse(data):
            coo = data.tocoo()
            return CompactTensor(
                data=coo.data.copy(),
                row=coo.row.astype(np.int64),
                col=coo.col.astype(np.int64),
                param_idx=np.zeros(len(coo.data), dtype=np.int64),
                m=data.shape[0],
                n=data.shape[1],
                param_size=1
            )
        else:
            data = np.asarray(data)
            # Reshape to target shape (e.g., (1, n) for row vector)
            if data.ndim == 1:
                data = data.reshape(target_shape)
            nz = data != 0
            row, col = np.where(nz)
            return CompactTensor(
                data=data[nz].flatten(),
                row=row.astype(np.int64),
                col=col.astype(np.int64),
                param_idx=np.zeros(nz.sum(), dtype=np.int64),
                m=data.shape[0],
                n=data.shape[1],
                param_size=1
            )

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

    def div(self, lin_op, view: COOTensorView) -> COOTensorView:
        """
        Division by constant: x / d.

        Note: div currently doesn't support parameters in divisor.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=True)
        assert is_param_free_lhs, "div doesn't support parametrized divisor"

        # Get reciprocal values
        lhs_compact = self._make_compact_from_sparse(lhs)
        # Invert the data
        recip_data = np.reciprocal(lhs_compact.data, dtype=float)
        lhs_recip = CompactTensor(
            data=recip_data,
            row=lhs_compact.row,
            col=lhs_compact.col,
            param_idx=lhs_compact.param_idx,
            m=lhs_compact.m,
            n=lhs_compact.n,
            param_size=lhs_compact.param_size
        )

        def div_func(compact, p):
            return compact_mul_elem(compact, lhs_recip)

        view.accumulate_over_variables(div_func, is_param_free_function=True)
        return view

    def mul_elem(self, lin_op, view: COOTensorView) -> COOTensorView:
        """
        Element-wise multiplication: x * d.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin_op.data, view, column=True)

        if is_param_free_lhs:
            lhs_compact = self._make_compact_from_sparse(lhs)

            def func(compact, p):
                return compact_mul_elem(compact, lhs_compact)

            view.accumulate_over_variables(func, is_param_free_function=True)
        else:
            # Parametrized mul_elem
            def parametrized_mul_elem(rhs_compact):
                result = {}
                for param_id, lhs_compact in lhs.items():
                    lhs_ct = self._make_compact_from_sparse(lhs_compact, param_id)
                    result[param_id] = compact_mul_elem(lhs_ct, rhs_compact)
                return result

            new_tensor = {}
            for var_id, var_tensor in view.tensor.items():
                const_compact = var_tensor[Constant.ID.value]
                new_tensor[var_id] = parametrized_mul_elem(const_compact)

            return view.create_new_tensor_view(
                view.variable_ids, new_tensor, is_parameter_free=False
            )

        return view

    @staticmethod
    def promote(lin_op, view: COOTensorView) -> COOTensorView:
        """Promote scalar by repeating."""
        num_entries = int(np.prod(lin_op.shape))
        rows = np.zeros(num_entries, dtype=np.int64)
        view.select_rows(rows)
        return view

    @staticmethod
    def broadcast_to(lin_op, view: COOTensorView) -> COOTensorView:
        """Broadcast to shape."""
        broadcast_shape = lin_op.shape
        original_shape = lin_op.args[0].shape
        rows = np.arange(np.prod(original_shape, dtype=int)).reshape(original_shape, order='F')
        rows = np.broadcast_to(rows, broadcast_shape).flatten(order="F")
        view.select_rows(rows.astype(np.int64))
        return view

    # index: use base class implementation (via select_rows)

    def diag_vec(self, lin_op, view: COOTensorView) -> COOTensorView:
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

            return CompactTensor(
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

    def _make_compact_from_sparse(self, matrix, param_id=None):
        """Convert sparse matrix to CompactTensor."""
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

        return CompactTensor(
            data=coo.data.copy(),
            row=row.astype(np.int64),
            col=coo.col.astype(np.int64),
            param_idx=param_idx.astype(np.int64),
            m=m,
            n=coo.shape[1],
            param_size=p
        )

    # vstack, hstack, concatenate: use base class implementations

    def rmul(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Right multiplication: x @ B."""
        raise NotImplementedError("rmul not yet implemented in COOCanonBackend")

    @staticmethod
    def reshape_constant_data(constant_data: dict, lin_op_shape: tuple) -> dict:
        """Reshape constant data from column format to required shape."""
        result = {}
        for k, v in constant_data.items():
            if isinstance(v, CompactTensor):
                new_m = lin_op_shape[0] if len(lin_op_shape) > 0 else 1
                new_n = lin_op_shape[1] if len(lin_op_shape) > 1 else 1
                result[k] = compact_reshape(v, new_m * new_n, 1)
            else:
                result[k] = v.reshape((-1, 1), order='F') if hasattr(v, 'reshape') else v
        return result

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """Returns a function to extend and shift a CompactTensor."""
        def stack_func(compact, p):
            return CompactTensor(
                data=compact.data.copy(),
                row=compact.row + offset,
                col=compact.col.copy(),
                param_idx=compact.param_idx.copy(),
                m=total_rows,
                n=compact.n,
                param_size=compact.param_size
            )
        return stack_func

    def conv(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Discrete convolution."""
        raise NotImplementedError("conv not yet implemented in COOCanonBackend")

    def kron_r(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Kronecker product kron(a, x)."""
        raise NotImplementedError("kron_r not yet implemented in COOCanonBackend")

    def kron_l(self, lin_op, view: COOTensorView) -> COOTensorView:
        """Kronecker product kron(x, a)."""
        raise NotImplementedError("kron_l not yet implemented in COOCanonBackend")

    def trace(self, lin_op, view: COOTensorView) -> COOTensorView:
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
