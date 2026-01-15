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

from typing import Any, Callable, Iterator, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops import LinOp
from cvxpy.lin_ops.backends.base import (
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
    get_nd_matmul_dims,
    get_nd_rmul_dims,
    is_batch_varying,
)


def _build_interleaved_matrix_mul(
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

    Note: Batch broadcasting is handled symbolically in MulExpression, so
    const_shape and var_shape batch dimensions are guaranteed to match here.
    """
    B = int(np.prod(const_shape[:-2]))
    m_dim = const_shape[-2]
    k_dim = const_shape[-1]
    n = var_shape[-1]

    # Reshape to (B, m, k) in Fortran order
    const_flat = np.reshape(const_data, (B, m_dim, k_dim), order="F")

    # Build interleaved matrix using vectorized operations
    b_indices = np.arange(B)
    i_indices = np.arange(m_dim)
    r_indices = np.arange(k_dim)

    # bb, ii, rr are grids of indices corresponding to b, i, r in the formula below
    bb, ii, rr = np.meshgrid(b_indices, i_indices, r_indices, indexing="ij")

    rows = (bb + B * ii).ravel()
    cols = (bb + B * rr).ravel()
    data = const_flat.ravel()

    M_interleaved = sp.csr_array((data, (rows, cols)), shape=(B * m_dim, B * k_dim))

    if n > 1:
        return sp.kron(sp.eye_array(n, format="csr"), M_interleaved)
    return M_interleaved


def _apply_nd_kron_structure_mul(
    lhs: sp.sparray,
    batch_size: int,
    n: int,
) -> sp.sparray:
    """
    Apply ND Kronecker structure I_n ⊗ C ⊗ I_batch for mul (C @ X).

    For ND matmul C @ X where C is 2D (m, k) and X has shape (..., k, n):
    vec(C @ X) = (I_n ⊗ C ⊗ I_batch) @ vec(X)
    """
    if batch_size > 1:
        inner = sp.kron(lhs, sp.eye_array(batch_size, format="csr"))
    else:
        inner = lhs

    if n > 1:
        return sp.kron(sp.eye_array(n, format="csr"), inner)
    return inner


def _expand_parametric_slices_mul(
    stacked_matrix: sp.sparray,
    param_size: int,
    batch_size: int,
    n: int,
) -> Iterator[sp.sparray]:
    """
    Generator yielding expanded slices for parametric ND mul (C @ X).

    For a stacked parameter matrix of shape (param_size * m, k), extracts each
    (m, k) slice and applies I_n ⊗ C ⊗ I_batch structure.
    """
    m = stacked_matrix.shape[0] // param_size

    for slice_idx in range(param_size):
        slice_matrix = stacked_matrix[slice_idx * m:(slice_idx + 1) * m, :]
        yield _apply_nd_kron_structure_mul(slice_matrix, batch_size, n)


def _build_interleaved_matrix_rmul(
    const_data: np.ndarray,
    const_shape: Tuple[int, ...],
    var_shape: Tuple[int, ...],
) -> sp.csr_array:
    """
    Build the interleaved matrix for batch-varying rmul case.

    For X (..., m, k) @ C (..., k, n), builds M_interleaved ⊗ I_m where:
    M_interleaved[b + B*j, b + B*r] = C[b, r, j]

    This captures the Fortran-order vectorization where batch indices are interleaved:
    - X[b, i, r] is at index b + B*i + B*m*r
    - result[b, i, j] is at index b + B*i + B*m*j

    Note: Batch broadcasting is handled symbolically in MulExpression, so
    const_shape and var_shape batch dimensions are guaranteed to match here.
    """
    B = int(np.prod(const_shape[:-2]))
    k = const_shape[-2]
    n = const_shape[-1]
    m = var_shape[-2]

    # Reshape to (B, k, n) in Fortran order
    const_flat = np.reshape(const_data, (B, k, n), order="F")

    # Build interleaved indices for all (b, r, j) combinations
    b_indices = np.arange(B)
    r_indices = np.arange(k)
    j_indices = np.arange(n)

    bb, rr, jj = np.meshgrid(b_indices, r_indices, j_indices, indexing="ij")
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
        new_data, new_row, new_col = data, base_row, base_col

    return sp.csr_array((new_data, (new_row, new_col)), shape=(B * m * n, B * m * k))


def _apply_nd_kron_structure_rmul(
    rhs: sp.sparray,
    batch_size: int,
    m: int,
) -> sp.sparray:
    """
    Apply ND Kronecker structure for rmul: C.T tensor I_{batch*m}.

    For ND rmul X @ C where C is 2D (k, n) and X has shape (..., m, k):
    vec(X @ C) = (C.T tensor I_{batch*m}) @ vec(X)
    """
    rhs_t = rhs.T
    reps = batch_size * m
    if reps > 1:
        return sp.kron(rhs_t, sp.eye_array(reps, format="csr"))
    return rhs_t


def _expand_parametric_slices_rmul(
    stacked_matrix: sp.sparray,
    param_size: int,
    batch_size: int,
    m: int,
) -> Iterator[sp.sparray]:
    """
    Generator yielding expanded slices for parametric ND rmul.

    For a stacked parameter matrix of shape (param_size * k, n), extracts each
    (k, n) slice and applies C.T tensor I_{batch*m} structure.
    """
    k = stacked_matrix.shape[0] // param_size

    for slice_idx in range(param_size):
        slice_matrix = stacked_matrix[slice_idx * k:(slice_idx + 1) * k, :]
        yield _apply_nd_kron_structure_rmul(slice_matrix, batch_size, m)


class SciPyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        This is calculated by dividing the totals rows of the tensor by the
        number of parameter slices.
        """
        if self.tensor is not None:
            for param_dict in self.tensor.values():
                for param_id, param_mat in param_dict.items():
                    return param_mat.shape[0] // self.param_to_size[param_id]
        else:
            raise ValueError('Tensor cannot be None')

    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        This function iterates through all the tensor data and constructs their
        respective representation in COO format. The row data is adjusted according
        to the position of each element within a parameter slice. The parameter_offset
        finds which slice the original row indices belong to before applying the column
        offset.
        """
        assert self.tensor is not None
        shape = (total_rows, self.var_length + 1)
        tensor_representations = []
        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, parameter_matrix in variable_tensor.items():
                p = self.param_to_size[parameter_id]
                m = parameter_matrix.shape[0] // p
                coo_repr = parameter_matrix.tocoo(copy=False)
                tensor_representations.append(TensorRepresentation(
                    coo_repr.data,
                    (coo_repr.row % m) + row_offset,
                    coo_repr.col + self.id_to_col[variable_id],
                    coo_repr.row // m + self.param_to_col[parameter_id],
                    shape=shape
                ))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor. If there are multiple parameters 'p',
        we must select the same 'rows' from each parameter slice. This is done by
        introducing an offset of size 'm' for every parameter.
        """
        def func(x, p):
            if p == 1:
                return x[rows, :]
            else:
                m = x.shape[0] // p
                return x[np.tile(rows, p) + np.repeat(np.arange(p) * m, len(rows)), :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
        self.tensor = {var_id: {k: func(v, self.param_to_size[k])
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> SciPyTensorView:
        """
        Create new SciPyTensorView with same shape information as self,
        but new tensor data.
        """
        return SciPyTensorView(variable_ids, tensor, is_parameter_free,
                               self.param_size_plus_one, self.id_to_col,
                               self.param_to_size, self.param_to_col,
                               self.var_length)

    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, sp.spmatrix]) \
            -> dict[int, sp.spmatrix]:
        """
        Apply 'func' to each slice of the parameter representation.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: sp.spmatrix, b: sp.spmatrix) -> sp.spmatrix:
        """
        Apply element-wise summation on two sparse matrices.
        """
        return a + b

    @staticmethod
    def tensor_type():
        """
        The tensor representation of the stacked slices backend is one big
        sparse matrix instead of smaller sparse matrices in a list.
        """
        return (sp.sparray, sp.spmatrix)


class SciPyCanonBackend(PythonCanonBackend):
    @staticmethod
    def get_constant_data_from_const(lin_op: LinOp) -> sp.csr_array:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        data = [[lin_op.data]] if np.isscalar(lin_op.data) else lin_op.data
        constant = sp.csr_array(data)
        assert constant.shape == lin_op.shape
        return constant

    def reshape_constant_data(self, constant_data: dict[int, sp.csc_array],
                              lin_op_shape: tuple[int, int]) \
            -> dict[int, sp.csc_array]:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        the stacked slices of the tensor 'v' according to the lin_op_shape argument.
        """
        return {k: self._reshape_single_constant_tensor(v, lin_op_shape, self.param_to_size[k])
                for k, v in constant_data.items()}

    @staticmethod
    def _reshape_single_constant_tensor(v: sp.csc_array, lin_op_shape: tuple[int, int],
                                        param_size: int) -> sp.csc_array:
        """
        Reshape constant data from column format to matrix format.

        Dispatches to specialized functions based on whether data is parametric.

        Parameters
        ----------
        v : sparse column vector of shape (N, 1)
        lin_op_shape : (m, k) target matrix dimensions
        param_size : number of parameter slices (1 = non-parametric)

        Returns
        -------
        Reshaped sparse matrix of shape (param_size * m, k)
        """
        if param_size == 1:
            return SciPyCanonBackend._reshape_nonparametric(v, lin_op_shape)
        else:
            return SciPyCanonBackend._reshape_parametric(v, lin_op_shape, param_size)

    @staticmethod
    def _reshape_nonparametric(v: sp.csc_array, lin_op_shape: tuple[int, int]) -> sp.csc_array:
        """
        Reshape non-parametric constant data from column to matrix format.

        For a column vector of shape (p * m * k, 1), reshape to (p * m, k)
        where p is the number of copies (from broadcast operations).

        The reshaping follows Fortran (column-major) order.
        """
        assert v.shape[1] == 1
        m, k = lin_op_shape

        coo = v.tocoo()
        data, stacked_rows = coo.data, coo.row

        # p = number of copies from broadcast operations (often > 1 for ND matmul,
        # but can also occur from element-wise ops with broadcasting)
        p = np.prod(v.shape) // np.prod(lin_op_shape)
        slice_size = v.shape[0] // p

        # Extract slice index and position within slice
        slices, rows = np.divmod(stacked_rows, slice_size)

        # Reshape: linear index -> (row, col) in column-major order
        new_cols, new_rows = np.divmod(rows, m)
        new_rows = slices * m + new_rows

        new_stacked_shape = (p * m, k)
        return sp.csc_array((data, (new_rows, new_cols)), shape=new_stacked_shape)

    @staticmethod
    def _reshape_parametric(v: sp.csc_array, lin_op_shape: tuple[int, int],
                            param_size: int) -> sp.csc_array:
        """
        Reshape parametric constant data from column to matrix format.

        For parametric data, entries may be duplicated by broadcast operations.
        We deduplicate and compute positions based on param_idx.

        The param_idx encodes which parameter value each entry corresponds to.
        After broadcast_to, entries are duplicated but param_idx stays the same.
        We keep only the first occurrence of each param_idx.

        Parameters
        ----------
        v : sparse column of shape (broadcast_size * param_size, 1)
        lin_op_shape : (m, k) target matrix dimensions
        param_size : number of parameter values

        Returns
        -------
        Sparse matrix of shape (param_size * m, k)
        """
        assert v.shape[1] == 1
        m, k = lin_op_shape

        coo = v.tocoo()
        data, stacked_rows = coo.data, coo.row

        # Each param slice has slice_size rows; param_idx = stacked_row // slice_size
        slice_size = v.shape[0] // param_size
        param_idx = stacked_rows // slice_size

        # Deduplicate: broadcast creates copies with same param_idx.
        # For a param with param_size=12, param_idx should be 0-11 exactly once.
        # If param_idx=5 appears 3 times, broadcast created duplicates - keep first only.
        unique_param_idx, first_occurrence = np.unique(param_idx, return_index=True)

        # Position in (m, k) matrix: column-major order
        new_rows = unique_param_idx % m
        new_cols = unique_param_idx // m

        # In stacked format, offset by slice: unique_param_idx is also the slice index
        stacked_new_rows = unique_param_idx * m + new_rows

        new_stacked_shape = (param_size * m, k)
        return sp.csc_array(
            (data[first_occurrence], (stacked_new_rows, new_cols)),
            shape=new_stacked_shape
        )

    def get_empty_view(self) -> SciPyTensorView:
        """
        Returns an empty view of the corresponding SciPyTensorView subclass,
        coupling the SciPyCanonBackend subclass with the SciPyTensorView subclass.
        """
        return SciPyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                              self.param_to_size, self.param_to_col,
                                              self.var_length)

    def _mul_kronecker(
        self,
        lhs: sp.sparray,
        const_shape: Tuple[int, ...],
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        2D constant @ ND variable: use Kronecker structure I_n ⊗ C ⊗ I_batch.

        This is the most common case: a 2D constant matrix multiplies each
        batch element of an ND variable.
        """
        batch_size, n, _ = get_nd_matmul_dims(const_shape, var_shape)
        stacked_lhs = _apply_nd_kron_structure_mul(lhs, batch_size, n)

        def func(x, p):
            if p == 1:
                return (stacked_lhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye_array(p, format="csc"), stacked_lhs) @ x).tocsc()

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def _mul_interleaved(
        self,
        const: LinOp,
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        Batch-varying constant @ variable: use interleaved matrix structure.

        When the constant has batch dimensions (e.g., C(B,m,k) @ X(B,k,n)),
        each batch element uses a different constant matrix.
        """
        const_shape = const.shape
        # Raw data access is intentional: batch-varying constants are never parametric
        # (they're concrete arrays with explicit batch dims), so we skip get_constant_data.
        stacked_lhs = _build_interleaved_matrix_mul(const.data, const_shape, var_shape)

        def func(x, p):
            if p == 1:
                return (stacked_lhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye_array(p, format="csc"), stacked_lhs) @ x).tocsc()

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def _mul_parametric_lhs(
        self,
        lhs: dict,
        const_shape: Tuple[int, ...],
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        Parametric constant @ variable: expand each parameter slice with Kronecker.

        When the constant depends on Parameters, we expand each parameter slice
        separately using the Kronecker structure.
        """
        batch_size, n, _ = get_nd_matmul_dims(const_shape, var_shape)

        # Expand each parameter slice and stack
        stacked_lhs = {
            param_id: sp.vstack(
                list(_expand_parametric_slices_mul(v, self.param_to_size[param_id], batch_size, n)),
                format="csc"
            )
            for param_id, v in lhs.items()
        }

        def parametrized_mul(x):
            return {k: v @ x for k, v in stacked_lhs.items()}

        return view.accumulate_over_variables(parametrized_mul, is_param_free_function=False)

    def _rmul_kronecker(
        self,
        rhs: sp.sparray,
        const_shape: Tuple[int, ...],
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        ND variable @ 2D constant: use Kronecker structure C.T ⊗ I_{batch*m}.

        This is the most common case: a 2D constant matrix multiplies each
        batch element of an ND variable from the right.
        """
        batch_size, m, n, _ = get_nd_rmul_dims(var_shape, const_shape)
        stacked_rhs = _apply_nd_kron_structure_rmul(rhs, batch_size, m)

        def func(x, p):
            if p == 1:
                return (stacked_rhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye_array(p, format="csc"), stacked_rhs) @ x).tocsc()

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def _rmul_interleaved(
        self,
        const: LinOp,
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        Batch-varying variable @ constant: use interleaved matrix structure.

        When the constant has batch dimensions (e.g., X(B,m,k) @ C(B,k,n)),
        each batch element uses a different constant matrix.
        """
        const_shape = const.shape
        # Raw data access is intentional: batch-varying constants are never parametric
        stacked_rhs = _build_interleaved_matrix_rmul(const.data, const_shape, var_shape)

        def func(x, p):
            if p == 1:
                return (stacked_rhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye_array(p, format="csc"), stacked_rhs) @ x).tocsc()

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def _rmul_parametric_rhs(
        self,
        rhs: dict,
        const_shape: Tuple[int, ...],
        var_shape: Tuple[int, ...],
        view: SciPyTensorView,
    ) -> SciPyTensorView:
        """
        Parametric variable @ constant: expand each parameter slice with Kronecker.

        When the constant depends on Parameters, we expand each parameter slice
        separately using the Kronecker structure for rmul.
        """
        batch_size, m, n, _ = get_nd_rmul_dims(var_shape, const_shape)

        # Expand each parameter slice and stack
        stacked_rhs = {
            param_id: sp.vstack(
                list(_expand_parametric_slices_rmul(
                    v, self.param_to_size[param_id], batch_size, m)),
                format="csc"
            )
            for param_id, v in rhs.items()
        }

        def parametrized_rmul(x):
            return {k: v @ x for k, v in stacked_rhs.items()}

        return view.accumulate_over_variables(parametrized_rmul, is_param_free_function=False)

    def mul(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Matrix multiply: C @ X where C is constant/parametric, X is variable.

        Three cases (explicitly branched for clarity):
        1. Parametric LHS: Parameter @ Variable
        2. Batch-varying constant: C(B,m,k) @ X(B,k,n) - uses interleaved matrix
        3. 2D constant: C(m,k) @ X(B,k,n) - uses Kronecker I_n ⊗ C ⊗ I_B
        """
        const = lin.data
        var_shape = lin.args[0].shape
        const_shape = const.shape

        # Get constant data - this also tells us if it's parametric
        # Compute 2D target shape (last 2 dims for ND, row vector for 1D)
        target = const_shape[-2:] if len(const_shape) >= 2 else (1, const_shape[0])
        lhs, is_param_free = self.get_constant_data(const, view, target_shape=target)

        if not is_param_free:
            # Case 1: Parametric LHS
            return self._mul_parametric_lhs(lhs, const_shape, var_shape, view)
        elif is_batch_varying(const_shape):
            # Case 2: Batch-varying constant
            return self._mul_interleaved(const, var_shape, view)
        else:
            # Case 3: 2D constant (or trivial batch dims like (1, m, k))
            return self._mul_kronecker(lhs, const_shape, var_shape, view)

    @staticmethod
    def promote(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Promote view by repeating along axis 0 (rows).
        """
        num_entries = int(np.prod(lin.shape))
        rows = np.zeros(num_entries).astype(int)
        view.select_rows(rows)
        return view

    @staticmethod
    def broadcast_to(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Broadcast view by calling np.broadcast_to on the rows and indexing the view.
        """
        broadcast_shape = lin.shape
        original_shape = lin.args[0].shape
        rows = np.arange(np.prod(original_shape, dtype=int)).reshape(original_shape, order='F')
        rows = np.broadcast_to(rows, broadcast_shape).flatten(order="F")
        view.select_rows(rows)
        return view

    def mul_elem(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        When dealing with parametrized constant data, we need to repeat the variable tensor p times
        and stack them vertically to ensure shape compatibility for elementwise multiplication
        with the parametrized expression.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, target_shape=None)
        if is_param_free_lhs:
            def func(x, p):
                if p == 1:
                    return lhs.multiply(x)
                else:
                    new_lhs = sp.vstack([lhs] * p)
                    return new_lhs.multiply(x)
        else:
            def parametrized_mul(x):
                return {k: v.multiply(sp.vstack([x] * self.param_to_size[k]))
                        for k, v in lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def sum_entries(self, _lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view, return the sum of the representation
        on the row axis, ie: (sum(A, axis=axis), sum(b, axis=axis)).
        Here, since the slices are stacked, we sum over the rows corresponding
        to the same slice.

        Note: we form the sparse output directly using _get_sum_row_indices and
        column indices in column-major order.
        """
        sum_coeff_matrix = self._get_sum_coeff_matrix

        def func(x, p):
            shape = tuple(_lin.args[0].shape)
            axis, _ = _lin.data
            if p == 1:
                if axis is None:
                    return sp.csr_array(x.sum(axis=0).reshape(1, x.shape[1]))
                else:
                    A = sum_coeff_matrix(shape=shape, axis=axis)
                    return A @ x
            else:
                m = x.shape[0] // p
                if axis is None:
                    return (sp.kron(sp.eye_array(p, format="csc"), np.ones((1, m))) @ x).tocsc()
                else:
                    A = sum_coeff_matrix(shape=shape, axis=axis)
                    return (sp.kron(sp.eye_array(p, format="csc"), A) @ x).tocsc()

        view.apply_all(func)
        return view

    def _get_sum_row_indices(self, shape: tuple, axis: tuple) -> np.ndarray:
        """
        Internal function that computes the row indices corresponding to the sum
        along a specified axis.
        """
        out_axes = np.isin(range(len(shape)), axis, invert=True)
        out_idx = np.indices(shape)[out_axes]
        out_dims = np.array(shape)[out_axes]
        row_idx = np.ravel_multi_index(out_idx, dims=out_dims, order='F')
        return row_idx.flatten(order='F')

    def _get_sum_coeff_matrix(self, shape: tuple, axis: tuple) -> sp.csr_array:
        """
        Internal function that computes the sum coefficient matrix for a given shape and axis.
        """
        axis = axis if isinstance(axis, tuple) else (axis,)
        n = np.prod(shape, dtype=int)
        d = np.prod([shape[i] for i in axis], dtype=int)
        row_idx = self._get_sum_row_indices(shape, axis)
        col_idx = np.arange(n)
        A = sp.csr_array((np.ones(n), (row_idx, col_idx)), shape=(n // d, n))
        return A

    def div(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns).
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data.

        Note: div currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, target_shape=None)
        assert is_param_free_lhs
        # dtype is important here, will do integer division if data is of dtype "int" otherwise.
        lhs.data = np.reciprocal(lhs.data, dtype=float)

        def div_func(x, p):
            if p == 1:
                return lhs.multiply(x)
            else:
                new_lhs = sp.vstack([lhs] * p)
                return new_lhs.multiply(x)

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x, p):
            shape = list(x.shape)
            shape[0] = int(total_rows * p)
            x = x.tocoo()
            x_slice, x_row = np.divmod(x.row, x.shape[0] // p)
            if k == 0:
                new_rows = x_row * (rows + 1)
            elif k > 0:
                new_rows = x_row * (rows + 1) + rows * k
            else:
                new_rows = x_row * (rows + 1) - k
            new_rows = (new_rows + x_slice * total_rows).astype(int)
            return sp.csc_array((x.data, (new_rows, x.col)), shape)

        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 0.
        """
        def stack_func(tensor, p):
            coo_repr = tensor.tocoo()
            m = coo_repr.shape[0] // p
            slices = coo_repr.row // m
            new_rows = (coo_repr.row + (slices + 1) * offset)
            new_rows = new_rows + slices * (total_rows - m - offset).astype(int)
            return sp.csc_array((coo_repr.data, (new_rows, coo_repr.col)),
                                shape=(int(total_rows * p), tensor.shape[1]))

        return stack_func

    def rmul(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Right multiplication: X @ C where X is variable, C is constant/parametric.

        For X @ C where X is (m, k) variable and C is (k, n) constant:
        vec(X @ C) = (C.T tensor I_m) @ vec(X)

        Three cases (explicitly branched for clarity):
        1. Parametric RHS: X @ Parameter
        2. Batch-varying constant: X(B,m,k) @ C(B,k,n) - uses interleaved matrix
        3. 2D constant: X(m,k) @ C(k,n) - uses Kronecker C.T ⊗ I_m
        """
        const = lin.data
        var_shape = lin.args[0].shape
        const_shape = const.shape

        # Get constant data - this also tells us if it's parametric
        # Compute 2D target shape (last 2 dims for ND, column vector for 1D)
        target = const_shape[-2:] if len(const_shape) >= 2 else (const_shape[0], 1)
        rhs, is_param_free = self.get_constant_data(const, view, target_shape=target)

        if not is_param_free:
            # Case 1: Parametric RHS
            return self._rmul_parametric_rhs(rhs, const_shape, var_shape, view)
        elif is_batch_varying(const_shape):
            # Case 2: Batch-varying constant
            return self._rmul_interleaved(const, var_shape, view)
        else:
            # Case 3: 2D constant (or trivial batch dims like (1, k, n))
            return self._rmul_kronecker(rhs, const_shape, var_shape, view)

    def _transpose_stacked(self, v: sp.csc_array, param_id: int) -> sp.csc_array:
        """
        Given v, which is a stacked matrix of shape (p * n, m), transpose each slice of v,
        returning a stacked matrix of shape (p * m, n).
        """
        old_shape = (v.shape[0] // self.param_to_size[param_id], v.shape[1])
        p = v.shape[0] // old_shape[0]
        new_shape = (old_shape[1], old_shape[0])
        new_stacked_shape = (p * new_shape[0], new_shape[1])

        v = v.tocoo()
        data, rows, cols = v.data, v.row, v.col
        slices, rows = np.divmod(rows, old_shape[0])

        new_rows = cols + slices * new_shape[0]
        new_cols = rows

        return sp.csc_array((data, (new_rows, new_cols)), shape=new_stacked_shape)

    def _stacked_kron_l(self, lhs: dict[int, list[sp.csc_array]], reps: int) \
            -> sp.csc_array:
        """
        Given a stacked lhs with the following entries:
        [[a11, a12],
         [a21, a22],
         ...
        Apply the Kronecker product with the identity matrix of size reps
        (kron(lhs, eye(reps))) to each slice.
        """
        res = dict()
        for param_id, v in lhs.items():
            self.param_to_size[param_id]
            coo = v.tocoo()
            data, rows, cols = coo.data, coo.row, coo.col
            new_rows = np.repeat(rows * reps, reps) + np.tile(np.arange(reps), len(rows))
            new_cols = np.repeat(cols * reps, reps) + np.tile(np.arange(reps), len(cols))
            new_data = np.repeat(data, reps)
            new_shape = (v.shape[0] * reps, v.shape[1] * reps)
            res[param_id] = sp.csc_array(
                (new_data, (new_rows, new_cols)), shape=new_shape)
        return res

    @staticmethod
    def trace(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        Apply kron(eye(p), lhs) to deal with parametrized expressions.
        """
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])

        data = np.ones(len(indices))
        idx = (np.zeros(len(indices)), indices.astype(int))
        lhs = sp.csr_array((data, idx), shape=(1, np.prod(shape)))

        def func(x, p) -> sp.csc_array:
            if p == 1:
                return (lhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye_array(p, format="csc"), lhs) @ x).tocsc()

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        # Compute target shape (2D shape, or row vector for 1D, or (1,1) for 0D)
        data_shape = lin.data.shape
        if len(data_shape) == 2:
            target = data_shape
        elif len(data_shape) == 1:
            target = (1, data_shape[0])
        else:  # 0D scalar
            target = (1, 1)
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, target_shape=target)
        assert is_param_free_lhs, \
            "SciPy backend does not support parametrized left operand for conv."
        assert lhs.ndim == 2

        if len(lin.data.shape) == 1:
            lhs = lhs.T

        rows = lin.shape[0]
        cols = lin.args[0].shape[0] if len(lin.args[0].shape) > 0 else 1

        lhs = lhs.tocoo()
        nonzeros = lhs.nnz

        row_idx = (np.tile(lhs.row, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        col_idx = (np.tile(lhs.col, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        data = np.tile(lhs.data, cols)

        lhs = sp.csr_array((data, (row_idx, col_idx)), shape=(rows, cols))

        def func(x, p):
            assert p == 1, \
                "SciPy backend does not support parametrized right operand for conv."
            return lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_r currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, target_shape=None)
        assert is_param_free_lhs, \
            "SciPy backend does not support parametrized left operand for kron_r."
        assert lhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape

        row_idx = self._get_kron_row_indices(lin.data.shape, rhs_shape)

        def func(x, p):
            assert p == 1, \
                "SciPy backend does not support parametrized right operand for kron_r."
            assert x.ndim == 2
            kron_res = sp.kron(lhs, x).tocsr()
            kron_res = kron_res[row_idx, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_l currently doesn't support parameters.
        """
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, target_shape=None)
        assert is_param_free_rhs, \
            "SciPy backend does not support parametrized right operand for kron_l."
        assert rhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape

        row_idx = self._get_kron_row_indices(lhs_shape, lin.data.shape)

        def func(x, p):
            assert p == 1, \
                "SciPy backend does not support parametrized left operand for kron_l."
            assert x.ndim == 2
            kron_res = sp.kron(x, rhs).tocsr()
            kron_res = kron_res[row_idx, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> \
            dict[int, dict[int, sp.csc_array]]:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        This function returns eye(n) in csc format.
        """
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: sp.eye_array(n, format="csc")}}

    def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> \
            dict[int, dict[int, sp.csr_array]]:
        """
        Returns tensor of constant node as a column vector.
        This function reshapes the data and converts it to csc format.
        """
        if isinstance(data, np.ndarray):
            # Slightly faster compared to reshaping after casting
            tensor = sp.csr_array(data.reshape((-1, 1), order="F"))
        else:
            tensor = sp.coo_matrix(data).reshape((-1, 1), order="F").tocsr()
        return {Constant.ID.value: {Constant.ID.value: tensor}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> \
            dict[int, dict[int, sp.csc_array]]:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        This function returns eye(n).flatten() in csc format.
        """
        assert parameter_id != Constant.ID
        param_size = self.param_to_size[parameter_id]
        shape = (int(np.prod(shape) * param_size), 1)
        arg = np.ones(param_size), (np.arange(param_size) + np.arange(param_size) * param_size,
                                    np.zeros(param_size))
        param_vec = sp.csc_array(arg, shape)
        return {Constant.ID.value: {parameter_id: param_vec}}
