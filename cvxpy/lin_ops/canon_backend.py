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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve

from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
    NUMPY_CANON_BACKEND,
    RUST_CANON_BACKEND,
    SCIPY_CANON_BACKEND,
    STACKED_SLICES_BACKEND,
)


class Constant(Enum):
    ID = -1


@dataclass
class TensorRepresentation:
    """
    Sparse representation of a 3D Tensor. Semantically similar to COO format, with one extra
    dimension. Here, 'row' is axis 0, 'col' axis 1, and 'parameter_offset' axis 2.
    """
    data: np.ndarray
    row: np.ndarray
    col: np.ndarray
    parameter_offset: np.ndarray

    @classmethod
    def combine(cls, tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Concatenates the row, col, parameter_offset, and data fields of a list of
        TensorRepresentations.
        """
        data, row, col, parameter_offset = np.array([]), np.array([]), np.array([]), np.array([])
        # Appending to numpy arrays vs. appending to lists and casting to array at the end was
        # faster for relevant dimensions in our testing.
        for t in tensors:
            data = np.append(data, t.data)
            row = np.append(row, t.row)
            col = np.append(col, t.col)
            parameter_offset = np.append(parameter_offset, t.parameter_offset)
        return cls(data, row, col, parameter_offset)

    def __eq__(self, other: TensorRepresentation) -> bool:
        return isinstance(other, TensorRepresentation) and \
            np.all(self.data == other.data) and \
            np.all(self.row == other.row) and \
            np.all(self.col == other.col) and \
            np.all(self.parameter_offset == other.parameter_offset)


class CanonBackend(ABC):
    def __init__(self, id_to_col: dict[int, int], param_to_size: dict[int, int],
                 param_to_col: dict[int, int], param_size_plus_one: int, var_length: int):
        """
        CanonBackend handles the compilation from LinOp trees to a final sparse tensor through its
        subclasses.

        Parameters
        ----------
        id_to_col: mapping of variable id to column offset in A.
        param_to_size: mapping of parameter id to the corresponding number of elements.
        param_to_col: mapping of parameter id to the offset in axis 2.
        param_size_plus_one: integer representing shape[2], i.e., the number of slices along axis 2
                             plus_one refers to the non-parametrized slice of the tensor.
        var_length: number of columns in A.
        """
        self.param_size_plus_one = param_size_plus_one
        self.id_to_col = id_to_col
        self.param_to_size = param_to_size
        self.param_to_col = param_to_col
        self.var_length = var_length

    @classmethod
    def get_backend(cls, backend_name: str, *args, **kwargs) -> CanonBackend:
        """
        Map the name of a subclass and its initializing arguments to an instance of the subclass.

        Parameters
        ----------
        backend_name: key pointing to the subclass.
        args: Arguments required to initialize the subclass.

        Returns
        -------
        Initialized CanonBackend subclass.
        """
        backends = {
            NUMPY_CANON_BACKEND: NumpyCanonBackend,
            SCIPY_CANON_BACKEND: ScipyCanonBackend,
            STACKED_SLICES_BACKEND: StackedSlicesBackend,
            RUST_CANON_BACKEND: RustCanonBackend
        }
        return backends[backend_name](*args, **kwargs)

    @abstractmethod
    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_matrix:
        """
        Main function called from canonInterface.
        Given a list of LinOp trees, each representing a constraint (or the objective), get the
        [A b] Tensor for each, stack them and return the result reshaped as a 2D sp.csc_matrix
        of shape (total_rows * (var_length + 1)), param_size_plus_one)

        Parameters
        ----------
        lin_ops: list of linOp trees.

        Returns
        -------
        2D sp.csc_matrix representing the constraints (or the objective).
        """
        pass  # noqa


class PythonCanonBackend(CanonBackend):
    """
    Each tensor has 3 dimensions. The first one is the parameter axis, the second one is the rows
    and the third one is the variable columns.

    For example:
    - A new variable of size n has shape (1, n, n)
    - A new parameter of size n has shape (n, n, 1)
    - A new constant of size n has shape (1, n, 1)
    """

    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_matrix:
        self.id_to_col[-1] = self.var_length

        constraint_res = []
        row_offset = 0
        for lin_op in lin_ops:
            lin_op_rows = np.prod(lin_op.shape)
            empty_view = self.get_empty_view()
            lin_op_tensor = self.process_constraint(lin_op, empty_view)
            constraint_res.append((lin_op_tensor.get_tensor_representation(row_offset)))
            row_offset += lin_op_rows
        tensor_res = self.concatenate_tensors(constraint_res)

        self.id_to_col.pop(-1)
        return self.reshape_tensors(tensor_res, row_offset)

    def process_constraint(self, lin_op: LinOp, empty_view: TensorView) -> TensorView:
        """
        Depth-first parsing of a linOp node.

        Parameters
        ----------
        lin_op: a node in the linOp tree.
        empty_view: TensorView used to create tensors for leaf nodes.

        Returns
        -------
        The processed node as a TensorView.
        """

        # Leaf nodes
        if lin_op.type == "variable":
            assert isinstance(lin_op.data, int)
            assert len(lin_op.shape) in {0, 1, 2}
            variable_tensor = self.get_variable_tensor(lin_op.shape, lin_op.data)
            return empty_view.create_new_tensor_view({lin_op.data}, variable_tensor,
                                                     is_parameter_free=True)
        elif lin_op.type in {"scalar_const", "dense_const", "sparse_const"}:
            data_tensor = self.get_data_tensor(lin_op.data)
            return empty_view.create_new_tensor_view({Constant.ID.value}, data_tensor,
                                                     is_parameter_free=True)
        elif lin_op.type == "param":
            param_tensor = self.get_param_tensor(lin_op.shape, lin_op.data)
            return empty_view.create_new_tensor_view({Constant.ID.value}, param_tensor,
                                                     is_parameter_free=False)

        # Internal nodes
        else:
            func = self.get_func(lin_op.type)
            if lin_op.type in {"vstack", "hstack"}:
                return func(lin_op, empty_view)

            res = None
            for arg in lin_op.args:
                arg_coeff = self.process_constraint(arg, empty_view)
                arg_res = func(lin_op, arg_coeff)
                if res is None:
                    res = arg_res
                else:
                    res += arg_res
            assert res is not None
            return res

    def get_constant_data(self, lin_op: LinOp, view: TensorView, column: bool) \
            -> tuple[sp.csr_matrix, bool]:
        """
        Extract the constant data from a LinOp node. In most cases, lin_op will be of
        type "*_const" or "param", but can handle arbitrary types.
        """
        constant_view = self.process_constraint(lin_op, view)
        assert constant_view.variable_ids == {Constant.ID.value}
        constant_data = constant_view.tensor[Constant.ID.value]
        if not column and len(lin_op.shape) >= 1:
            # constant_view has the data stored in column format.
            # Some operations (like mul) do not require column format, so we need to reshape
            # according to lin_op.shape.
            lin_op_shape = lin_op.shape if len(lin_op.shape) == 2 else [1, lin_op.shape[0]]
            constant_data = self.reshape_constant_data(constant_data, lin_op_shape)

        data_to_return = constant_data[Constant.ID.value] if constant_view.is_parameter_free \
            else constant_data
        return data_to_return, constant_view.is_parameter_free

    @staticmethod
    @abstractmethod
    def reshape_constant_data(constant_data: Any, lin_op_shape: tuple[int, int]) -> Any:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format
        """
        pass  # noqa

    @staticmethod
    def concatenate_tensors(tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Takes list of tensors and stacks them along axis 0 (rows).
        """
        return TensorRepresentation.combine(tensors)

    def reshape_tensors(self, tensor: TensorRepresentation, total_rows: int) -> sp.csc_matrix:
        """
           Reshape into 2D scipy coo-matrix in column-major order and transpose.
        """
        rows = (tensor.col.astype(np.int64) * np.int64(total_rows) + tensor.row.astype(np.int64))
        cols = tensor.parameter_offset.astype(np.int64)
        shape = (np.int64(total_rows) * np.int64(self.var_length + 1), self.param_size_plus_one)
        return sp.csc_matrix((tensor.data, (rows, cols)), shape=shape)

    @abstractmethod
    def get_empty_view(self) -> TensorView:
        """
        Returns an empty view of the corresponding TensorView subclass, coupling the CanonBackend
        subclass with the TensorView subclass.
        """
        pass  # noqa

    def get_func(self, func_name: str) -> Callable:
        """
        Map the name of a function as given by the linOp to the implementation.

        Parameters
        ----------
        func_name: The name of the function.

        Returns
        -------
        The function implementation.
        """
        mapping = {
            "sum": self.sum_op,
            "mul": self.mul,
            "promote": self.promote,
            "neg": self.neg,
            "mul_elem": self.mul_elem,
            "sum_entries": self.sum_entries,
            "div": self.div,
            "reshape": self.reshape,
            "index": self.index,
            "diag_vec": self.diag_vec,
            "hstack": self.hstack,
            "vstack": self.vstack,
            "transpose": self.transpose,
            "upper_tri": self.upper_tri,
            "diag_mat": self.diag_mat,
            "rmul": self.rmul,
            "trace": self.trace,
            "conv": self.conv,
            "kron_l": self.kron_l,
            "kron_r": self.kron_r,
        }
        return mapping[func_name]

    @staticmethod
    def sum_op(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Sum (along axis 1) is implicit in Ax+b, so it is a NOOP.
        """
        return view

    @staticmethod
    def reshape(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Reshaping only changes the shape attribute of the LinOp, so it is a NOOP.
        """
        return view

    @abstractmethod
    def mul(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Multiply view with constant data from the left
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def promote(lin: LinOp, view: TensorView) -> TensorView:
        """
        Promote view by repeating along axis 0 (rows)
        """
        pass  # noqa

    @staticmethod
    def neg(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, return (-A, -b).
        """

        def func(x, _p=1):
            return -x

        view.apply_all(func)
        return view

    @abstractmethod
    def mul_elem(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        d is broadcasted along dimension 1 (columns)
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def sum_entries(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, return (sum(A,axis=0), sum(b, axis=0))
        """
        pass  # noqa

    @abstractmethod
    def div(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns)
        """
        pass  # noqa

    @staticmethod
    def index(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, select the rows corresponding to the elements of the expression being
        indexed.
        """
        indices = [np.arange(s.start, s.stop, s.step) for s in lin.data]
        if len(indices) == 1:
            rows = indices[0]
        elif len(indices) == 2:
            rows = np.add.outer(indices[0], indices[1] * lin.args[0].shape[0]).flatten(order="F")
        else:
            raise ValueError
        view.select_rows(rows)
        return view

    @staticmethod
    @abstractmethod
    def diag_vec(lin: LinOp, view: TensorView) -> TensorView:
        """
        Diagonal vector to matrix. Given (A, b) in with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 0.
        """
        pass  # noqa

    def hstack(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given views (A0,b0), (A1,b1),..., (An,bn), stack all tensors along axis 0,
        i.e., return
        (A0, b0)
         A1, b1
         ...
         An, bn.
        """
        offset = 0
        total_rows = sum(np.prod(arg.shape) for arg in lin.args)
        res = None
        for arg in lin.args:
            arg_view = self.process_constraint(arg, view)
            func = self.get_stack_func(total_rows, offset)
            arg_view.apply_all(func)
            offset += np.prod(arg.shape)
            if res is None:
                res = arg_view
            else:
                res += arg_view
        assert res is not None
        return res

    def vstack(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given views (A0,b0), (A1,b1),..., (An,bn), first, stack them along axis 0 via hstack.
        Then, permute the rows of the resulting tensor to be consistent with stacking the arguments
        vertically instead of horizontally.
        """
        view = self.hstack(lin, view)
        offset = 0
        indices = []
        for arg in lin.args:
            arg_rows = np.prod(arg.shape)
            indices.append(np.arange(arg_rows).reshape(arg.shape, order="F") + offset)
            offset += arg_rows
        order = np.vstack(indices).flatten(order="F").astype(int)
        view.select_rows(order)
        return view

    @staticmethod
    def transpose(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, permute the rows such that they correspond to the transposed
        expression.
        """
        rows = np.arange(np.prod(lin.shape)).reshape(lin.shape).flatten(order="F")
        view.select_rows(rows)
        return view

    @staticmethod
    def upper_tri(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, select the rows corresponding to the elements above the diagonal
        in the original expression.
        Note: The diagonal itself is not included.
        """
        indices = np.arange(np.prod(lin.args[0].shape)).reshape(lin.args[0].shape, order="F")
        triu_indices = indices[np.triu_indices_from(indices, k=1)]
        view.select_rows(triu_indices)
        return view

    @staticmethod
    def diag_mat(lin: LinOp, view: TensorView) -> TensorView:
        """
        Diagonal matrix to vector. Given (A, b) in view, select the rows corresponding to the
        elements above the diagonal in the original expression.
        """
        rows = lin.shape[0]
        k = lin.data
        original_rows = rows + abs(k)
        if k == 0:
            diag_indices = np.arange(rows) * (rows + 1)
        elif k > 0:
            diag_indices = np.arange(rows) * (original_rows + 1) + original_rows * k
        else:
            diag_indices = np.arange(rows) * (original_rows + 1) - k
        view.select_rows(diag_indices)
        return view

    @abstractmethod
    def rmul(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Multiply view with constant data from the right
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def trace(lin: LinOp, view: TensorView) -> TensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        """
        pass  # noqa

    @abstractmethod
    def conv(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column.
        """
        pass  # noqa

    @abstractmethod
    def kron_r(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).
        """
        pass  # noqa

    @abstractmethod
    def kron_l(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).
        """
        pass  # noqa

    @abstractmethod
    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> Any:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n i the number of
        entries of the variable.
        """
        pass  # noqa

    @abstractmethod
    def get_data_tensor(self, data: Any) -> Any:
        """
        Returns tensor of constant node as a column vector.
        """
        pass  # noqa

    @abstractmethod
    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> Any:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n i the number
        of entries of the parameter.
        """
        pass  # noqa


class RustCanonBackend(CanonBackend):
    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_matrix:
        import cvxpy_rust
        self.id_to_col[-1] = self.var_length
        (data, (row, col), shape) = cvxpy_rust.build_matrix(lin_ops,
                                                            self.param_size_plus_one,
                                                            self.id_to_col,
                                                            self.param_to_size,
                                                            self.param_to_col,
                                                            self.var_length)
        self.id_to_col.pop(-1)
        return sp.csc_matrix((data, (row, col)), shape)


class ScipyCanonBackend(PythonCanonBackend):
    @staticmethod
    def reshape_constant_data(constant_data: dict[int, sp.csr_matrix],
                              lin_op_shape: tuple[int, int]) \
            -> dict[int, sp.csr_matrix]:
        return {k: [v_i.reshape(lin_op_shape, order="F").tocsr()
                    for v_i in v] for k, v in constant_data.items()}

    def get_empty_view(self) -> ScipyTensorView:
        return ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                              self.param_to_size, self.param_to_col,
                                              self.var_length)

    def mul(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)

        if isinstance(lhs, dict):
            reps = view.rows // next(iter(lhs.values()))[0].shape[-1]
            eye = sp.eye(reps, format="csr")
            stacked_lhs = {k: [sp.kron(eye, v_i).tocsr() for v_i in v] for k, v in lhs.items()}

            def parametrized_mul(x):
                assert len(x) == 1
                return {k: [(v_i @ x[0]).tocsr() for v_i in v] for k, v in stacked_lhs.items()}

            func = parametrized_mul
        else:
            assert len(lhs) == 1
            reps = view.rows // lhs[0].shape[-1]
            stacked_lhs = (sp.kron(sp.eye(reps, format="csr"), lhs[0]))

            def func(x):
                return stacked_lhs.tocsr() @ x

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def promote(lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        num_entries = int(np.prod(lin.shape))

        def func(x):
            # Fast way of repeating sparse matrix along axis 0
            # See comment in https://stackoverflow.com/a/50759652
            return x[np.zeros(num_entries, dtype=int), :]

        view.apply_all(func)
        return view

    def mul_elem(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        if isinstance(lhs, dict):
            def parametrized_mul(x):
                assert len(x) == 1
                return {k: [v_i.multiply(x[0]).tocsr() for v_i in v] for k, v in lhs.items()}

            func = parametrized_mul
        else:
            def func(x):
                return lhs[0].multiply(x)
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def sum_entries(_lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        def func(x):
            return sp.csr_matrix(x.sum(axis=0))

        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]

        # dtype is important here, will do integer division if data is of dtype "int" otherwise.
        lhs.data = np.reciprocal(lhs.data, dtype=float)

        def div_func(x):
            return lhs.multiply(x)

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x):
            shape = list(x.shape)
            shape[0] = total_rows
            x = x.tocoo()
            if k == 0:
                new_rows = (x.row * (rows + 1)).astype(int)
            elif k > 0:
                new_rows = (x.row * (rows + 1) + rows * k).astype(int)
            else:
                new_rows = (x.row * (rows + 1) - k).astype(int)
            return sp.csr_matrix((x.data, (new_rows, x.col)), shape)

        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        def stack_func(tensor):
            coo_repr = tensor.tocoo()
            new_rows = (coo_repr.row + offset).astype(int)
            return sp.csr_matrix((coo_repr.data, (new_rows, coo_repr.col)),
                                 shape=(int(total_rows), tensor.shape[1]))

        return stack_func

    def rmul(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        # Note that even though this is rmul, we still use "lhs", as is implemented via a
        # multiplication from the left in this function.
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)

        arg_cols = lin.args[0].shape[0] if len(lin.args[0].shape) == 1 else lin.args[0].shape[1]

        if isinstance(lhs, dict):

            lhs_shape = next(iter(lhs.values()))[0].shape

            if len(lin.data.shape) == 1 and arg_cols != lhs_shape[0]:
                # Example: (n,n) @ (n,), we need to interpret the rhs as a column vector,
                # but it is a row vector by default, so we need to transpose
                lhs = {k: [v_i.T for v_i in v] for k, v in lhs.items()}
                lhs_shape = next(iter(lhs.values()))[0].shape

            reps = view.rows // lhs_shape[0]
            eye = sp.eye(reps, format="csr")

            stacked_lhs = {k: [sp.kron(v_i.T, eye) for v_i in v] for k, v in lhs.items()}

            def parametrized_mul(x):
                assert len(x) == 1
                return {k: [(v_i @ x[0]).tocsr() for v_i in v] for k, v in stacked_lhs.items()}

            func = parametrized_mul
        else:
            assert len(lhs) == 1
            lhs = lhs[0]
            if len(lin.data.shape) == 1 and arg_cols != lhs.shape[0]:
                # Example: (n,n) @ (n,), we need to interpret the rhs as a column vector,
                # but it is a row vector by default, so we need to transpose
                lhs = lhs.T
            reps = view.rows // lhs.shape[0]
            stacked_lhs = sp.kron(lhs.T, sp.eye(reps, format="csr"))

            def func(x):
                return stacked_lhs @ x
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def trace(lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])

        data = np.ones(len(indices))
        idx = (np.zeros(len(indices)), indices.astype(int))
        lhs = sp.csr_matrix((data, idx), shape=(1, np.prod(shape)))

        def func(x):
            return lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]
        assert lhs.ndim == 2

        if len(lin.data.shape) == 1:
            lhs = lhs.T

        rows = lin.shape[0]
        cols = lin.args[0].shape[0]
        nonzeros = lhs.shape[0]

        lhs = lhs.tocoo()
        row_idx = (np.tile(lhs.row, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        col_idx = (np.tile(lhs.col, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        data = np.tile(lhs.data, cols)

        lhs = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))

        def func(x):
            return lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]
        assert lhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape

        lhs_ones = np.ones(lin.data.shape)
        rhs_ones = np.ones(rhs_shape)

        lhs_arange = np.arange(np.prod(lin.data.shape)).reshape(lin.data.shape, order="F")
        rhs_arange = np.arange(np.prod(rhs_shape)).reshape(rhs_shape, order="F")

        row_indices = (np.kron(lhs_ones, rhs_arange) +
                       np.kron(lhs_arange, rhs_ones * np.prod(rhs_shape))) \
            .flatten(order="F").astype(int)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 2
            kron_res = sp.kron(lhs, x).tocsr()
            kron_res = kron_res[row_indices, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_rhs
        assert len(rhs) == 1
        rhs = rhs[0]
        assert rhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape

        rhs_ones = np.ones(lin.data.shape)
        lhs_ones = np.ones(lhs_shape)

        rhs_arange = np.arange(np.prod(lin.data.shape)).reshape(lin.data.shape, order="F")
        lhs_arange = np.arange(np.prod(lhs_shape)).reshape(lhs_shape, order="F")

        row_indices = (np.kron(lhs_ones, rhs_arange) +
                       np.kron(lhs_arange, rhs_ones * np.prod(lin.data.shape))) \
            .flatten(order="F").astype(int)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 2
            kron_res = sp.kron(x, rhs).tocsr()
            kron_res = kron_res[row_indices, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> \
            dict[int, dict[int, list[sp.csr_matrix]]]:
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: [sp.eye(n, format="csr")]}}

    def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> \
            dict[int, dict[int, list[sp.csr_matrix]]]:
        if isinstance(data, np.ndarray):
            # Slightly faster compared to reshaping after casting
            tensor = sp.csr_matrix(data.reshape((-1, 1), order="F"))
        else:
            tensor = sp.coo_matrix(data).reshape((-1, 1), order="F").tocsr()
        return {Constant.ID.value: {Constant.ID.value: [tensor]}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) \
            -> dict[int, dict[int, list[sp.csr_matrix]]]:
        assert parameter_id != Constant.ID
        shape = int(np.prod(shape))
        slices = []
        for idx in np.arange(shape):
            slices.append(sp.csr_matrix(((np.array([1.])), ((np.array([idx])),
                                                            (np.array([0])))), shape=(shape, 1)))
        return {Constant.ID.value: {parameter_id: slices}}


class NumpyCanonBackend(PythonCanonBackend):
    @staticmethod
    def reshape_constant_data(constant_data: dict[int, np.ndarray],
                              lin_op_shape: tuple[int, int]) \
            -> dict[int, np.ndarray]:
        return {k: v.reshape((v.shape[0], *lin_op_shape), order="F")
                for k, v in constant_data.items()}

    def get_empty_view(self) -> NumpyTensorView:
        return NumpyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                              self.param_to_size, self.param_to_col,
                                              self.var_length)

    def mul(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
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
    def promote(lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        num_entries = int(np.prod(lin.shape))

        def func(x):
            return np.tile(x, (1, num_entries, 1))

        view.apply_all(func)
        return view

    def mul_elem(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
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
    def sum_entries(_lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        def func(x):
            return x.sum(axis=1, keepdims=True)

        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert lhs.shape[0] == 1
        # dtype is important here, will do integer division if data is of dtype "int" otherwise.
        lhs = np.reciprocal(lhs, where=lhs != 0, dtype=float)

        def div_func(x):
            return lhs * x

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
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
        def stack_func(tensor):
            rows = tensor.shape[1]
            new_rows = (np.arange(rows) + offset).astype(int)
            matrix = np.zeros(shape=(tensor.shape[0], int(total_rows), tensor.shape[2]))
            matrix[:, new_rows, :] = tensor
            return matrix

        return stack_func

    def rmul(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        # Note that even though this is rmul, we still use "lhs", as is implemented via a
        # multiplication from the left in this function.
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
    def trace(lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])

        lhs = np.zeros(shape=(1, np.prod(shape)))
        lhs[0, indices] = 1

        def func(x):
            return lhs @ x

        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        assert is_param_free_lhs

        if len(lin.data.shape) == 1:
            lhs = np.swapaxes(lhs, -2, -1)

        def func(x):
            return convolve(lhs, x)

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]
        assert lhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape

        lhs_ones = np.ones(lin.data.shape)
        rhs_ones = np.ones(rhs_shape)

        lhs_arange = np.arange(np.prod(lin.data.shape)).reshape(lin.data.shape, order="F")
        rhs_arange = np.arange(np.prod(rhs_shape)).reshape(rhs_shape, order="F")

        row_indices = (np.kron(lhs_ones, rhs_arange) +
                       np.kron(lhs_arange, rhs_ones * np.prod(rhs_shape))) \
            .flatten(order="F").astype(int)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(lhs, x)
            kron_res = kron_res[:, row_indices, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: NumpyTensorView) -> NumpyTensorView:
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_rhs
        assert len(rhs) == 1
        rhs = rhs[0]
        assert rhs.ndim == 2

        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape

        rhs_ones = np.ones(lin.data.shape)
        lhs_ones = np.ones(lhs_shape)

        rhs_arange = np.arange(np.prod(lin.data.shape)).reshape(lin.data.shape, order="F")
        lhs_arange = np.arange(np.prod(lhs_shape)).reshape(lhs_shape, order="F")

        row_indices = (np.kron(lhs_ones, rhs_arange) +
                       np.kron(lhs_arange, rhs_ones * np.prod(lin.data.shape))) \
            .flatten(order="F").astype(int)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(x, rhs)
            kron_res = kron_res[:, row_indices, :]
            return kron_res

        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) \
            -> dict[int, dict[int, np.ndarray]]:
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: np.expand_dims(np.eye(n), axis=0)}}

    def get_data_tensor(self, data: np.ndarray) -> \
            dict[int, dict[int, np.ndarray]]:
        data = self._to_dense(data)
        tensor = data.reshape((-1, 1), order="F")
        return {Constant.ID.value: {Constant.ID.value: np.expand_dims(tensor, axis=0)}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) \
            -> dict[int, dict[int, np.ndarray]]:
        assert parameter_id != Constant.ID
        n = int(np.prod(shape))
        return {Constant.ID.value: {parameter_id: np.expand_dims(np.eye(n), axis=-1)}}

    @staticmethod
    def _to_dense(x):
        try:
            res = x.toarray()
        except AttributeError:
            res = x
        res = np.atleast_2d(res)
        return res


class StackedSlicesBackend(PythonCanonBackend):

    def reshape_constant_data(self, constant_data: dict[int, sp.csc_matrix],
                              new_shape: tuple[int, ...]) -> dict[int, sp.csc_matrix]:
        assert len(new_shape) == 2
        (m, n) = new_shape
        res = {}
        for param_id, param_vec in constant_data.items():
            p = self.param_to_size[param_id]
            old_shape = param_vec.shape
            coo = param_vec.tocoo(copy=False)
            data, stacked_rows, cols = coo.data, coo.row, coo.col
            slice, rows = np.divmod(stacked_rows, old_shape[0])

            element_position = cols * old_shape[0] + rows
            new_rows, new_cols = np.divmod(element_position, m)
            new_rows = slice * m + new_rows
            res[param_id] = sp.csc_matrix((data, (new_rows, new_cols)), shape=(m * p, n))
        return res

    def get_empty_view(self) -> TensorView:
        return StackedSlicesTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                      self.param_to_size, self.param_to_col,
                                                      self.var_length)

    def mul(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        """
        Multiply view with constant data from the left.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        if is_param_free_lhs:
            reps = view.rows // lhs.shape[-1]
            stacked_lhs = (sp.kron(sp.eye(reps, format="csc"), lhs))

            def func(x, _p):
                return stacked_lhs.tocsc() @ x
        else:
            reps = view.rows // next(iter(lhs.values())).shape[-1]
            eye = sp.eye(reps, format="csc")

            if reps > 1:
                stacked_lhs = {}
                for param_id, param_mat in lhs.items():
                    inds = np.arange(param_mat.shape[0])
                    sub_inds = np.split(inds, self.param_to_size[param_id])
                    stacked_lhs[param_id] = sp.vstack([sp.kron(eye, param_mat[sub_ind],
                                                               format='csc')
                                                       for sub_ind in sub_inds], format='csc')
            else:
                stacked_lhs = lhs

            def parametrized_mul(x):
                return {k: v @ x for k, v in stacked_lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def promote(lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        num_entries = int(np.prod(lin.shape))
        rows = np.zeros(num_entries)
        view.select_rows(rows)
        return view

    def mul_elem(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        When dealing with parametrized constant data, we need to repeat the variable tensor p times
        and stack them vertically to ensure shape compatibility for elementwise multiplication
        with the parametrized expression.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        if is_param_free_lhs:
            def func(x, _p):
                return lhs.multiply(x)
        else:
            reps = view.rows // next(iter(lhs.values())).shape[-1]

            def parametrized_mul(x):
                new_x = sp.vstack([x] * reps)
                return {k: v.multiply(new_x) for k, v in lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def sum_entries(_lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        def func(x, p):
            if p == 1:
                return sp.csr_matrix(x.sum(axis=0))
            else:
                m = x.shape[0] // p
                return (sp.kron(sp.eye(p, format="csr"), np.ones(m)) @ x).tocsr()

        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        # dtype is important here, will do integer division if data is of dtype "int" otherwise.
        lhs.data = np.reciprocal(lhs.data, dtype=float)

        def div_func(x, _p):
            return lhs.multiply(x)

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x, _p):
            shape = list(x.shape)
            shape[0] = total_rows
            x = x.tocoo()
            if k == 0:
                new_rows = (x.row * (rows + 1)).astype(int)
            elif k > 0:
                new_rows = (x.row * (rows + 1) + rows * k).astype(int)
            else:
                new_rows = (x.row * (rows + 1) - k).astype(int)
            return sp.csr_matrix((x.data, (new_rows, x.col)), shape)

        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        def stack_func(tensor, _p):
            coo_repr = tensor.tocoo()
            new_rows = (coo_repr.row + offset).astype(int)
            return sp.csr_matrix((coo_repr.data, (new_rows, coo_repr.col)),
                                 shape=(int(total_rows), tensor.shape[1]))

        return stack_func

    def rmul(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        """
        Multiply view with constant data from the right.
        """
        # Note that even though this is rmul, we still use "lhs", as is implemented via a
        # multiplication from the left in this function.
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)

        arg_cols = lin.args[0].shape[0] if len(lin.args[0].shape) == 1 else lin.args[0].shape[1]

        if is_param_free_lhs:
            if len(lin.data.shape) == 1 and arg_cols != lhs.shape[0]:
                lhs = lhs.T
            reps = view.rows // lhs.shape[0]
            stacked_lhs = sp.kron(lhs.T, sp.eye(reps, format="csr"))

            def func(x, _p):
                return stacked_lhs @ x
        else:
            lhs_shape = next(iter(lhs.values()))[0].shape

            if len(lin.data.shape) == 1 and arg_cols != lhs_shape[0]:
                # Example: (n,n) @ (n,), we need to interpret the rhs as a column vector,
                # but it is a row vector by default, so we need to transpose
                lhs = {k: v.T for k, v in lhs.items()}
                lhs_shape = next(iter(lhs.values())).shape

            reps = view.rows // lhs_shape[0]
            eye = sp.eye(reps, format="csr")

            stacked_lhs = {k: sp.kron(v.T, eye) for k, v in lhs.items()}

            def parametrized_mul(x):
                return {k: v @ x for k, v in stacked_lhs.items()}

            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def trace(lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        pass

    def conv(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        pass

    def kron_r(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        pass

    def kron_l(self, lin: LinOp, view: StackedSlicesTensorView) -> StackedSlicesTensorView:
        pass

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> \
            dict[int, dict[int, sp.csc_matrix]]:
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: sp.eye(n, format="csc")}}

    def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> \
            dict[int, dict[int, sp.csc_matrix]]:
        if isinstance(data, np.ndarray):
            # Slightly faster compared to reshaping after casting
            tensor = sp.csc_matrix(data.reshape((-1, 1), order="F"))
        else:
            tensor = sp.coo_matrix(data).reshape((-1, 1), order="F").tocsc()
        return {Constant.ID.value: {Constant.ID.value: tensor}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> \
            dict[int, dict[int, sp.csc_matrix]]:
        assert parameter_id != Constant.ID
        param_size = self.param_to_size[parameter_id]
        shape = (int(np.prod(shape) * param_size), 1)
        arg = np.ones(param_size), (np.arange(param_size) + np.arange(param_size) * param_size,
                                    np.zeros(param_size))
        param_vec = sp.csc_matrix(arg, shape)
        return {Constant.ID.value: {parameter_id: param_vec}}


class TensorView(ABC):
    """
    A TensorView represents the tensors for A and b, which are of shape
    rows x var_length x param_size_plus_one and rows x 1 x param_size_plus_one, respectively.
    The class facilitates the application of the CanonBackend functions.
    """

    def __init__(self,
                 variable_ids: set[int] | None,
                 tensor: Any,
                 is_parameter_free: bool,
                 param_size_plus_one: int,
                 id_to_col: dict[int, int],
                 param_to_size: dict[int, int],
                 param_to_col: dict[int, int],
                 var_length: int
                 ):
        self.variable_ids = variable_ids if variable_ids is not None else None
        self.tensor = tensor
        self.is_parameter_free = is_parameter_free

        # Constants
        self.param_size_plus_one = param_size_plus_one
        self.id_to_col = id_to_col
        self.param_to_size = param_to_size
        self.param_to_col = param_to_col
        self.var_length = var_length

    def __iadd__(self, other: TensorView) -> TensorView:
        assert isinstance(other, self.__class__)
        self.variable_ids = self.variable_ids | other.variable_ids
        self.tensor = self.combine_potentially_none(self.tensor, other.tensor)
        self.is_parameter_free = self.is_parameter_free and other.is_parameter_free
        return self

    @staticmethod
    @abstractmethod
    def combine_potentially_none(a: Any | None, b: Any | None) -> Any | None:
        """
        Adds the tensor a to b if they are both not none.
        If a (b) is not None but b (a) is None, returns a (b).
        Returns None if both a and b are None.
        """
        pass  # noqa

    @classmethod
    def get_empty_view(cls, param_size_plus_one: int, id_to_col: dict[int, int],
                       param_to_size: dict[int, int], param_to_col: dict[int, int],
                       var_length: int) \
            -> TensorView:
        """
        Return a TensorView that has shape information, but no data.
        """
        return cls(None, None, True, param_size_plus_one, id_to_col, param_to_size, param_to_col,
                   var_length)

    @staticmethod
    def is_constant_data(variable_ids: set[int]) -> bool:
        """
        Does the TensorView only contain constant data?
        """
        return variable_ids == {Constant.ID.value}

    @property
    @abstractmethod
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        """
        pass  # noqa

    @abstractmethod
    def get_tensor_representation(self, row_offset: int) -> TensorRepresentation:
        """
        Returns [A b].
        """
        pass  # noqa

    @abstractmethod
    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor.
        """
        pass  # noqa

    @abstractmethod
    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        """
        pass  # noqa

    @abstractmethod
    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any, is_parameter_free: bool) \
            -> TensorView:
        """
        Create new TensorView with same shape information as self, but new data.
        """
        pass  # noqa


class DictTensorView(TensorView, ABC):
    """
    The DictTensorView abstract class handles the dictionary aspect of the tensor representation,
    which is shared across multiple backends.
    The tensor is contained in the following data structure:
    `Dict[variable_id, Dict[parameter_id, tensor]]`, with the outer dict handling
    the variable offset, and the inner dict handling the parameter offset.
    Subclasses have to implement the implementation of the tensor, as well
    as the tensor operations.
    """

    def accumulate_over_variables(self, func: Callable, is_param_free_function: bool) \
            -> TensorView:
        """
        Apply 'func' to A and b.
        If 'func' is a parameter free function, then we can apply it to all parameter slices
        (including the slice that contains non-parameter constants).
        If 'func' is not a parameter free function, we only need to consider the parameter slice
        that contains the non-parameter constants, due to DPP rules.
        """
        for variable_id, tensor in self.tensor.items():
            self.tensor[variable_id] = self.apply_to_parameters(func, tensor) if \
                is_param_free_function else func(tensor[Constant.ID.value])

        self.is_parameter_free = self.is_parameter_free and is_param_free_function
        return self

    def combine_potentially_none(self, a: dict | None, b: dict | None) -> dict | None:
        if a is None and b is None:
            return None
        elif a is not None and b is None:
            return a
        elif a is None and b is not None:
            return b
        else:
            return self.add_dicts(a, b)

    @staticmethod
    @abstractmethod
    def add_tensors(a: Any, b: Any) -> Any:
        pass  # noqa

    @staticmethod
    @abstractmethod
    def tensor_type():
        pass  # noqa

    def add_dicts(self, a: dict, b: dict) -> dict:
        """
        Addition for dict-based tensors.
        """
        res = {}
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        intersect = keys_a & keys_b
        for key in intersect:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                res[key] = self.add_dicts(a[key], b[key])
            elif isinstance(a[key], self.tensor_type()) and isinstance(b[key], self.tensor_type()):
                assert len(a[key]) == len(b[key])
                res[key] = self.add_tensors(a[key], b[key])
            else:
                raise ValueError(f'Values must either be dicts or {self.tensor_type()}.')
        for key in keys_a - intersect:
            res[key] = a[key]
        for key in keys_b - intersect:
            res[key] = b[key]
        return res


class ScipyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        if self.tensor is not None:
            return next(iter(next(iter(self.tensor.values())).values()))[0].shape[0]
        else:
            raise ValueError

    def get_tensor_representation(self, row_offset: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        """
        assert self.tensor is not None
        tensor_representations = []
        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, parameter_tensor in variable_tensor.items():
                for param_slice_offset, matrix in enumerate(parameter_tensor):
                    coo_repr = matrix.tocoo(copy=False)
                    tensor_representations.append(TensorRepresentation(
                        coo_repr.data,
                        coo_repr.row + row_offset,
                        coo_repr.col + self.id_to_col[variable_id],
                        np.ones(coo_repr.nnz) * self.param_to_col[parameter_id] +
                        param_slice_offset,
                    ))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:

        def func(x):
            return x[rows, :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        self.tensor = {var_id: {k: [func(v_i).tocsr() for v_i in v]
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: dict,
                               is_parameter_free: bool) -> ScipyTensorView:
        return ScipyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one,
                               self.id_to_col, self.param_to_size, self.param_to_col,
                               self.var_length)

    @staticmethod
    def apply_to_parameters(func: Callable,
                            parameter_representation: dict[int, list[sp.csr_matrix]]) \
            -> dict[int, list[sp.csr_matrix]]:
        """
        Apply 'func' to each slice of the parameter representation.
        """
        return {k: [func(v_i).tocsr() for v_i in v] for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: list[sp.csr_matrix], b: list[sp.csr_matrix]) -> list[sp.csr_matrix]:
        return [a + b for a, b in zip(a, b)]

    def tensor_type(self):
        return list


class NumpyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        if self.tensor is not None:
            return next(iter(next(iter(self.tensor.values())).values())).shape[1]
        else:
            raise ValueError

    def get_tensor_representation(self, row_offset: int) -> TensorRepresentation:
        """
        CVXPY currently only supports usage of sparse matrices after the canonicalization.
        Therefore, we must return tensor representations in a (data, (row,col)) format.
        This could be changed once dense matrices are accepted.
        """
        assert self.tensor is not None
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
                ))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:

        def func(x):
            return x[:, rows, :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        self.tensor = {var_id: {k: func(v)
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> NumpyTensorView:
        return NumpyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one,
                               self.id_to_col, self.param_to_size, self.param_to_col,
                               self.var_length)

    @staticmethod
    def apply_to_parameters(func: Callable,
                            parameter_representation: dict[int, np.ndarray]) \
            -> dict[int, np.ndarray]:
        """
        Apply 'func' to each slice of the parameter representation.
        """
        return {k: func(v) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def tensor_type(self):
        return np.ndarray


class StackedSlicesTensorView(DictTensorView):
    """
    We need a way to be able to override the tensor indexing.
    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            i, j, k = indices
            if self.tensor is not None:
                return self.tensor[i][j][self.rows*k:self.rows*(k+1), :]
    """

    @property
    def rows(self) -> int:
        if self.tensor is not None:
            for var_id, param_dict in self.tensor.items():
                for param_id, param_mat in param_dict.items():
                    return param_mat.shape[0] // self.param_to_size[param_id]
        else:
            raise ValueError('Tensor cannot be None')

    def get_tensor_representation(self, row_offset: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        """
        assert self.tensor is not None
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
                    coo_repr.row // m,
                ))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        def func(x, p):
            if p == 1:
                return x[rows, :]
            else:
                m = x.shape[0] // p
                return x[np.tile(rows, p) + np.repeat(np.arange(p) * m, p), :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        self.tensor = {var_id: {k: func(v, self.param_to_size[k])
                                for k, v in parameter_repr.items()}
                       for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any,
                               is_parameter_free: bool) -> StackedSlicesTensorView:
        return StackedSlicesTensorView(variable_ids, tensor, is_parameter_free,
                                       self.param_size_plus_one, self.id_to_col,
                                       self.param_to_size, self.param_to_col,
                                       self.var_length)

    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, sp.spmatrix]) \
            -> dict[int, sp.spmatrix]:
        """
        Apply 'func' to each slice of the parameter representation.
        """
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: sp.spmatrix, b: sp.spmatrix) -> sp.spmatrix:
        return a + b

    @staticmethod
    def tensor_type():
        """
        The tensor representation of the stacked slices backend is one big
        sparse matrix instead of smaller sparse matrices in a list.
        """
        return sp.spmatrix
