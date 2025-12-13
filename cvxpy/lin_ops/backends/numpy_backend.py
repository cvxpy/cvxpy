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
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.signal import convolve

from cvxpy.lin_ops import LinOp
from cvxpy.lin_ops.backends import (
    Constant,
    DictTensorView,
    PythonCanonBackend,
    TensorRepresentation,
)


class NumPyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        This is the second dimension of the 3d tensor.
        """
        if self.tensor is not None:
            return next(iter(next(iter(self.tensor.values())).values())).shape[1]
        else:
            raise ValueError

    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        This function iterates through all the tensor data and constructs the
        respective representation in COO format. To obtain the data, the tensor must be
        flattened as it is not in a sparse format. The row and column indices are obtained
        with numpy tiling/repeating along with their respective offsets.

        Note: CVXPY currently only supports usage of sparse matrices after the canonicalization.
        Therefore, we must return tensor representations in a (data, (row,col)) format.
        This could be changed once dense matrices are accepted.
        """
        assert self.tensor is not None
        shape = (total_rows, self.var_length + 1)
        tensor_representations = []
        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, parameter_tensor in variable_tensor.items():
                param_size, rows, cols = parameter_tensor.shape
                tensor_representations.append(TensorRepresentation(
                    parameter_tensor.flatten(order='F'),
                    np.repeat(np.tile(np.arange(rows), cols), param_size) + row_offset,
                    np.repeat(np.repeat(np.arange(cols), rows), param_size)
                    + self.id_to_col[variable_id],
                    np.tile(np.arange(param_size), rows * cols) + self.param_to_col[parameter_id],
                    shape=shape
                ))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor.
        The rows of the 3d tensor are in axis=1, this function selects a subset
        of the original tensor.
        """
        def func(x):
            return x[:, rows, :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        The tensor functions in the NumPyBackend manipulate 3d arrays.
        Therefore, this function applies 'func' directly to the tensor 'v'.
        """
        self.tensor = {var_id: {k: func(v)
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> NumPyTensorView:
        """
        Create new NumPyTensorView with same shape information as self,
        but new tensor data.
        """
        return NumPyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one,
                               self.id_to_col, self.param_to_size, self.param_to_col,
                               self.var_length)

    @staticmethod
    def apply_to_parameters(func: Callable,
                            parameter_representation: dict[int, np.ndarray]) \
            -> dict[int, np.ndarray]:
        """
        Apply 'func' to the entire tensor of the parameter representation.
        """
        return {k: func(v) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Apply element-wise addition on two dense numpy arrays
        """
        return a + b

    @staticmethod
    def tensor_type():
        """
        The tensor is represented as a 3-dimensional dense numpy array
        """
        return np.ndarray


class NumPyCanonBackend(PythonCanonBackend):
    @staticmethod
    def get_constant_data_from_const(lin_op: LinOp) -> np.ndarray:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        constant = NumPyCanonBackend._to_dense(lin_op.data)
        assert constant.shape == lin_op.shape
        return constant

    @staticmethod
    def reshape_constant_data(constant_data: dict[int, np.ndarray],
                              lin_op_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        dimensions 1 and 2 of the tensor 'v' according to the lin_op_shape argument.
        """
        return {k: v.reshape((v.shape[0], *lin_op_shape), order="F")
                for k, v in constant_data.items()}

    def concatenate_tensors(self, tensors: list[TensorRepresentation]) \
            -> TensorRepresentation:
        """
        Takes list of tensors which have already been offset along axis 0 (rows) and
        combines them into a single tensor.
        """
        return TensorRepresentation.combine(tensors)

    def get_empty_view(self) -> NumPyTensorView:
        """
        Returns an empty view of the corresponding NumPyTensorView subclass,
        coupling the NumPyCanonBackend subclass with the NumPyTensorView subclass.
        """
        return NumPyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                              self.param_to_size, self.param_to_col,
                                              self.var_length)

    def mul(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the rhs.
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        if isinstance(lhs, dict):
            reps = view.rows // next(iter(lhs.values()))[0].shape[-1]
            stacked_lhs = {k: np.kron(np.eye(reps), v) for k, v in lhs.items()}

            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v @ x for k, v in stacked_lhs.items()}

            func = parametrized_mul
        else:
            assert isinstance(lhs, np.ndarray)
            reps = view.rows // lhs.shape[-1]
            stacked_lhs = np.kron(np.eye(reps), lhs)

            def func(x):
                return stacked_lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def promote(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Promote view by repeating along axis 1 (rows).
        """
        num_entries = int(np.prod(lin.shape))

        def func(x):
            return np.tile(x, (1, num_entries, 1))

        view.apply_all(func)
        return view

    @staticmethod
    def broadcast_to(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Broadcast view by calling np.broadcast_to on the rows and indexing the view.
        """
        broadcast_shape = lin.shape
        original_shape = lin.args[0].shape
        rows = np.arange(np.prod(original_shape, dtype=int)).reshape(original_shape, order='F')
        rows = np.broadcast_to(rows, broadcast_shape).flatten(order="F")
        view.select_rows(rows)
        return view

    def mul_elem(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        d is broadcasted along dimension 1 (columns).
        When the lhs is parametrized, multiply elementwise each slice of the tensor with the
        single, constant slice of the rhs.
        Otherwise, multiply elementwise the single slice of the tensor with each slice of the rhs.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        if isinstance(lhs, dict):
            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v * x for k, v in lhs.items()}

            func = parametrized_mul
        else:
            def func(x):
                return lhs * x
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def sum_entries(_lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view, return the sum of the representation
        on the row axis, ie: (sum(A, axis=axis), sum(b, axis=axis)).

        Note for new n-dimensional version: We now pass an axis parameter to the sum.
        The new implementation keeps the columns of the tensor fixed and reshapes the
        remaining dimensions in the original shape of the expression. The sum is then
        performed along the axis parameter. Finally, the tensor is reshaped back to the
        desired output shape.

        Example:
        # Suppose we want to sum a Variable(2,2,2)
        x = np.eye(8)
        out = x.reshape(2,2,2,8).sum(axis=axis).reshape(n // prod(shape[axis]),8)
        """
        def func(x):
            axis, _ = _lin.data
            if axis is None:
                return x.sum(axis=1, keepdims=True)
            else:
                shape = _lin.args[0].shape
                n = x.shape[-1]
                p = x.shape[0]
                if isinstance(axis, tuple):
                    d = np.prod([shape[i] for i in axis], dtype=int)
                    # adding offset of 1 to every axis because of param axis.
                    axis = tuple([a + 1 for a in axis])
                else:
                    d = shape[axis]
                    axis += 1
                x = x.reshape((p,)+shape+(n,), order='F').sum(axis=axis)
                return x.reshape((p, n//d, n), order='F')

        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns).
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data.

        Note: div currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert lhs.shape[0] == 1
        # dtype is important here, will do integer division if data is of dtype "int" otherwise.
        lhs = np.reciprocal(lhs, where=lhs != 0, dtype=float)

        def div_func(x):
            return lhs * x

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression.
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x):
            x_rows = x.shape[1]
            shape = list(x.shape)
            shape[1] = total_rows

            if k == 0:
                new_rows = np.arange(x_rows) * (rows + 1)
            elif k > 0:
                new_rows = np.arange(x_rows) * (rows + 1) + rows * k
            else:
                new_rows = np.arange(x_rows) * (rows + 1) - k
            matrix = np.zeros(shape)
            matrix[:, new_rows, :] = x
            return matrix

        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 1.
        """
        def stack_func(tensor):
            rows = tensor.shape[1]
            new_rows = (np.arange(rows) + offset).astype(int)
            matrix = np.zeros(shape=(tensor.shape[0], int(total_rows), tensor.shape[2]))
            matrix[:, new_rows, :] = tensor
            return matrix

        return stack_func

    def rmul(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
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
            lhs_rows = lhs.shape[-2]
            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = np.swapaxes(lhs, -2, -1)
            lhs_rows = lhs.shape[-2]
            reps = view.rows // lhs_rows
            lhs_transposed = np.swapaxes(lhs, -2, -1)
            stacked_lhs = np.kron(lhs_transposed, np.eye(reps))

            def func(x):
                return stacked_lhs @ x
        else:
            lhs_shape = next(iter(lhs.values()))[0].shape
            lhs_rows = lhs_shape[-2]
            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = {k: np.swapaxes(v, -2, -1) for k, v in lhs.items()}
                lhs_shape = next(iter(lhs.values()))[0].shape
            lhs_rows = lhs_shape[-2]
            reps = view.rows // lhs_rows
            stacked_lhs = {k: np.kron(np.swapaxes(v, -2, -1), np.eye(reps)) for k, v in lhs.items()}

            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v @ x for k, v in stacked_lhs.items()}

            func = parametrized_mul

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def trace(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        """
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])

        lhs = np.zeros(shape=(1, np.prod(shape)))
        lhs[0, indices] = 1

        def func(x):
            return lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        assert is_param_free_lhs

        if len(lin.data.shape) == 1:
            lhs = np.swapaxes(lhs, -2, -1)

        def func(x):
            return convolve(lhs, x)

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_r currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]
        assert lhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape

        row_idx = self._get_kron_row_indices(lin.data.shape, rhs_shape)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(lhs, x)
            kron_res = kron_res[:, row_idx, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_l currently doesn't support parameters.
        """
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_rhs
        assert len(rhs) == 1
        rhs = rhs[0]
        assert rhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape

        row_idx = self._get_kron_row_indices(lhs_shape, lin.data.shape)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(x, rhs)
            kron_res = kron_res[:, row_idx, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) \
            -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        This function expands the dimension of an identity matrix of size n on the parameter axis.
        """
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: np.expand_dims(np.eye(n), axis=0)}}

    def get_data_tensor(self, data: np.ndarray) -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of constant node as a column vector.
        This function expands the dimension of the column vector on the parameter axis.
        """
        data = self._to_dense(data)
        tensor = data.reshape((-1, 1), order="F")
        return {Constant.ID.value: {Constant.ID.value: np.expand_dims(tensor, axis=0)}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) \
            -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        This function expands the dimension of an identity matrix of size n on the column axis.
        """
        assert parameter_id != Constant.ID
        n = int(np.prod(shape))
        return {Constant.ID.value: {parameter_id: np.expand_dims(np.eye(n), axis=-1)}}

    @staticmethod
    def _to_dense(x):
        """
        Internal function that converts a sparse input to a dense numpy array.
        """
        try:
            res = x.toarray()
        except AttributeError:
            res = x
        res = np.atleast_2d(res)
        return res
