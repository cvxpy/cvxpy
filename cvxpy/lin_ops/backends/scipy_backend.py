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

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops import LinOp
from cvxpy.lin_ops.backends.base import (
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
)
from cvxpy.lin_ops.backends.nd_matmul_utils import (
    expand_lhs_for_nd_matmul,
    expand_parametric_slices,
    get_nd_matmul_dims,
)


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

    @staticmethod
    def reshape_constant_data(constant_data: dict[int, sp.csc_array],
                              lin_op_shape: tuple[int, int]) \
            -> dict[int, sp.csc_array]:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        the stacked slices of the tensor 'v' according to the lin_op_shape argument.
        """
        return {k: SciPyCanonBackend._reshape_single_constant_tensor(v, lin_op_shape)
                for k, v in constant_data.items()}

    @staticmethod
    def _reshape_single_constant_tensor(v: sp.csc_array, lin_op_shape: tuple[int, int]) \
            -> sp.csc_array:
        """
        Given v, which is a matrix of shape (p * lin_op_shape[0] * lin_op_shape[1], 1),
        reshape v into a matrix of shape (p * lin_op_shape[0], lin_op_shape[1]).
        """
        assert v.shape[1] == 1
        p = np.prod(v.shape) // np.prod(lin_op_shape)
        old_shape = (v.shape[0] // p, v.shape[1])

        coo = v.tocoo()
        data, stacked_rows = coo.data, coo.row
        slices, rows = np.divmod(stacked_rows, old_shape[0])

        new_cols, new_rows = np.divmod(rows, lin_op_shape[0])
        new_rows = slices * lin_op_shape[0] + new_rows

        new_stacked_shape = (p * lin_op_shape[0], lin_op_shape[1])
        return sp.csc_array((data, (new_rows, new_cols)), shape=new_stacked_shape)

    def get_empty_view(self) -> SciPyTensorView:
        """
        Returns an empty view of the corresponding SciPyTensorView subclass,
        coupling the SciPyCanonBackend subclass with the SciPyTensorView subclass.
        """
        return SciPyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                              self.param_to_size, self.param_to_col,
                                              self.var_length)

    def mul(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the rhs.
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.

        For ND arrays with 2D constant: I_n ⊗ C ⊗ I_batch
        For ND arrays with batch-varying constant: I_n ⊗ block_diag(C[0], ..., C[B-1])
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        var_shape = lin.args[0].shape
        const_shape = lin.data.shape

        if is_param_free_lhs:
            # Constant lhs: expand using unified helper
            stacked_lhs = expand_lhs_for_nd_matmul(lhs, lin.data.data, const_shape, var_shape)

            def func(x, p):
                if p == 1:
                    return (stacked_lhs @ x).tocsr()
                else:
                    return ((sp.kron(sp.eye_array(p, format="csc"), stacked_lhs)) @ x).tocsc()
        else:
            # Parametrized lhs: expand each param slice using unified helper
            batch_size, n, _ = get_nd_matmul_dims(const_shape, var_shape)
            stacked_lhs = {
                param_id: sp.vstack(
                    list(expand_parametric_slices(v, self.param_to_size[param_id], batch_size, n)),
                    format="csc"
                )
                for param_id, v in lhs.items()
            }

            def parametrized_mul(x):
                return {k: v @ x for k, v in stacked_lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

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
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
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
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
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
        Multiply view with constant data from the right.
        When the rhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the lhs.
        Otherwise, multiply the single slice of the tensor with each slice of the lhs.

        Note: Even though this is rmul, we still use "lhs", as is implemented via a
        multiplication from the left in this function.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)

        arg_cols = lin.args[0].shape[0] if len(lin.args[0].shape) == 1 else lin.args[0].shape[1]

        if is_param_free_lhs:

            if len(lin.data.shape) == 1 and arg_cols != lhs.shape[0]:
                lhs = lhs.T
            reps = view.rows // lhs.shape[0]
            if reps > 1:
                stacked_lhs = sp.kron(lhs.T, sp.eye_array(reps, format="csr"))
            else:
                stacked_lhs = lhs.T

            def func(x, p):
                if p == 1:
                    return (stacked_lhs @ x).tocsr()
                else:
                    return ((sp.kron(sp.eye_array(p, format="csc"), stacked_lhs)) @ x).tocsc()
        else:
            k, v = next(iter(lhs.items()))
            lhs_rows = v.shape[0] // self.param_to_size[k]

            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                # Example: (n,n) @ (n,), we need to interpret the rhs as a column vector,
                # but it is a row vector by default, so we need to transpose
                lhs = {k: self._transpose_stacked(v, k) for k, v in lhs.items()}
                k, v = next(iter(lhs.items()))
                lhs_rows = v.shape[0] // self.param_to_size[k]

            reps = view.rows // lhs_rows

            lhs = {k: self._transpose_stacked(v, k) for k, v in lhs.items()}

            if reps > 1:
                stacked_lhs = self._stacked_kron_l(lhs, reps)
            else:
                stacked_lhs = lhs

            def parametrized_mul(x):
                return {k: (v @ x).tocsc() for k, v in stacked_lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

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
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
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
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
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
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
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
