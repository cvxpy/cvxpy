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
from typing import Callable

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops import LinOp


class Constant(Enum):
    ID = -1


class CanonBackendName(Enum):
    SCIPY = "SCIPY"
    CPP = "CPP"


@dataclass
class TensorRepresentation:
    """
    Sparse representation of a 3D Tensor. Semantically similar to COO format, with one extra
    dimension. Here, 'row' is axis 0, 'col' axis 1, and 'parameter_offset' axis 2.
    """
    parameter_offset: np.ndarray
    row: np.ndarray
    col: np.ndarray
    data: np.ndarray

    @classmethod
    def combine(cls, tensors: list[TensorRepresentation]):
        """
        Concatenates the row, col, parameter_offset, and data fields of a list of
        TensorRepresentations.
        """
        parameter_offset, row, col, data = np.array([]), np.array([]), np.array([]), np.array([])
        for t in tensors:
            parameter_offset = np.append(parameter_offset, t.parameter_offset)
            row = np.append(row, t.row)
            col = np.append(col, t.col)
            data = np.append(data, t.data)
        return cls(parameter_offset, row, col, data)

    def __eq__(self, other):
        return isinstance(other, TensorRepresentation) and \
               np.all(self.parameter_offset == other.parameter_offset) and \
               np.all(self.row == other.row) and \
               np.all(self.col == other.col) and \
               np.all(self.data == other.data)


class CanonBackend(ABC):
    def __init__(self, id_to_col: dict[int, int], param_to_size: dict[int, int], param_to_col:
                 dict[int, int], param_size_plus_one: int, var_length: int):
        """
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
    def get_backend(cls, backend_name, *args):
        """
        Map the name of a subclass and its initializing arguments to an instance of the subclass.

        Parameters
        ----------
        backend_name: CanonBackendName of the subclass.
        args: Arguments required to initialize the subclass.

        Returns
        -------
        Initialized CanonBackend subclass.
        """
        backends = {
            CanonBackendName.SCIPY: ScipyCanonBackend,
        }
        return backends[backend_name](*args)

    def build_matrix(self, linOps: list[LinOp]) -> sp.coo_matrix:
        """
        Given a list of linOp trees, each representing a constraint (or the objective), get the
        [A b] Tensor for each, stack them and return the result reshaped as a 2D sp.coo_matrix
        Tensor.

        Parameters
        ----------
        linOps: list of linOp trees.

        Returns
        -------
        2D sp.coo_matrix representing the constraints (or the objective).
        """
        constraint_res = []
        offset = 0
        for lin_op in linOps:
            lin_op_rows = np.prod(lin_op.shape)
            empty_view = self.get_empty_view()
            lin_op_tensor = self.process_constraint(lin_op, empty_view)
            constraint_res.append((lin_op_tensor.get_A_b(), offset))
            offset += lin_op_rows
        tensor_res = self.concatenate_tensors(constraint_res)
        return self.reshape_tensors(tensor_res, offset)

    def process_constraint(self, lin_op: LinOp, empty_view: TensorView) -> TensorView:
        """
        Depth-first parsing of a linOp node.

        Parameters
        ----------
        lin_op: a node in the linOp tree.
        empty_view: TensorView used to create tensors for leaf nodes.

        Returns
        -------
        The processed constraint as a TensorView.
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
            -> tuple[np.ndarray, bool]:
        constant_view = self.process_constraint(lin_op, view)
        assert constant_view.variable_ids == {Constant.ID.value}
        constant_data = constant_view.constant_data
        if not column and len(lin_op.shape) >= 1:
            lin_op_shape = lin_op.shape if len(lin_op.shape) == 2 else [1, lin_op.shape[0]]
            new_shape = (self.param_size_plus_one, *lin_op_shape[-2:])
            constant_data = self.reshape_constant_data(constant_data, new_shape)

        data_to_return = constant_data[Constant.ID.value] if constant_view.is_parameter_free \
            else constant_data
        return data_to_return, constant_view.is_parameter_free

    @staticmethod
    @abstractmethod
    def reshape_constant_data(constant_data, new_shape: tuple[int, ...]):
        pass

    @abstractmethod
    def concatenate_tensors(self, tensors: list[tuple[TensorView, int]]) -> TensorView:
        """
        Takes list of tensors and stacks them along axis 0.
        """

    @abstractmethod
    def reshape_tensors(self, tensor: TensorView, total_rows: int) -> sp.coo_matrix:
        """
        Reshape into 2D scipy coo-matrix in column-major order and transpose.
        """

    @abstractmethod
    def get_empty_view(self) -> TensorView:
        pass

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
        pass

    @staticmethod
    @abstractmethod
    def promote(lin: LinOp, view: TensorView) -> TensorView:
        pass

    @staticmethod
    def neg(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b), return (-A, -b).
        """

        def func(x):
            return -x

        view.apply_all(func)
        return view

    @abstractmethod
    def mul_elem(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @staticmethod
    @abstractmethod
    def sum_entries(_lin: LinOp, view: TensorView) -> TensorView:
        pass

    @abstractmethod
    def div(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @staticmethod
    def index(lin: LinOp, view: TensorView) -> TensorView:
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
        pass

    @staticmethod
    @abstractmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        pass

    def hstack(self, lin: LinOp, view: TensorView) -> TensorView:
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
        rows = np.arange(np.prod(lin.shape)).reshape(lin.shape).flatten(order="F")
        view.select_rows(rows)
        return view

    @staticmethod
    def upper_tri(lin: LinOp, view: TensorView) -> TensorView:
        indices = np.arange(np.prod(lin.args[0].shape)).reshape(lin.args[0].shape, order="F")
        triu_indices = indices[np.triu_indices_from(indices, k=1)]
        view.select_rows(triu_indices)
        return view

    @staticmethod
    def diag_mat(lin: LinOp, view: TensorView) -> TensorView:
        # diagonal matrix to vector
        rows = lin.shape[0]
        diag_indices = np.arange(rows) * rows + np.arange(rows)
        view.select_rows(diag_indices)
        return view

    @abstractmethod
    def rmul(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @staticmethod
    @abstractmethod
    def trace(lin: LinOp, view: TensorView) -> TensorView:
        pass

    @abstractmethod
    def conv(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @abstractmethod
    def kron_r(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @abstractmethod
    def kron_l(self, lin: LinOp, view: TensorView) -> TensorView:
        pass

    @abstractmethod
    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int):
        pass

    @abstractmethod
    def get_data_tensor(self, data):
        pass

    @abstractmethod
    def get_param_tensor(self, shape, parameter_id):
        pass


class ScipyCanonBackend(CanonBackend):

    @staticmethod
    def reshape_constant_data(constant_data: dict[int, sp.csr_matrix], new_shape: tuple[int, ...]):
        return {k: [v_i.reshape(new_shape[1:], order="F")
                    for v_i in v] for k, v in constant_data.items()}

    def concatenate_tensors(self, tensors: list[tuple[TensorRepresentation, int]]):
        for tensor, row_offset in tensors:
            tensor.row += row_offset
        return TensorRepresentation.combine([t[0] for t in tensors])

    def reshape_tensors(self, tensor: TensorRepresentation, total_rows: int) -> sp.csc_matrix:
        # Windows uses int32 by default at time of writing, so we need to enforce int64 here
        rows = (tensor.col.astype(np.int64) * np.int64(total_rows) + tensor.row.astype(np.int64))
        cols = tensor.parameter_offset.astype(np.int64)
        shape = (np.int64(total_rows) * np.int64(self.var_length + 1), self.param_size_plus_one)
        return sp.csc_matrix((tensor.data, (rows, cols)), shape=shape)

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

        # dtype is important here, will do integer division if data is of dtype int otherwise.
        lhs.data = np.reciprocal(lhs.data, dtype=float)

        def div_func(x):
            return lhs.multiply(x)

        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: ScipyTensorView) -> ScipyTensorView:
        # vector to diagonal matrix
        assert lin.shape[0] == lin.shape[1]
        rows = lin.shape[0]
        total_rows = int(np.prod(lin.shape))

        def func(x):
            shape = list(x.shape)
            shape[0] = total_rows
            x = x.tocoo()
            new_rows = (x.row * rows + x.row).astype(int)
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

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int):
        assert variable_id != Constant.ID
        shape = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: [sp.eye(shape, format="csr")]}}

    def get_data_tensor(self, data):
        # TODO: Can this be made faster?
        if isinstance(data, np.ndarray):
            # Seems slightly faster
            tensor = sp.csr_matrix(data.reshape((-1, 1), order="F"))
        else:
            tensor = sp.coo_matrix(data).reshape((-1, 1), order="F").tocsr()
        return {Constant.ID.value: [tensor]}

    def get_param_tensor(self, shape, parameter_id):
        assert parameter_id != Constant.ID
        shape = int(np.prod(shape))
        slices = []
        for idx in np.arange(shape):
            slices.append(sp.csr_matrix(((np.array([1.])), ((np.array([idx])),
                                                            (np.array([0])))), shape=(shape, 1)))
        return {parameter_id: slices}


class TensorView(ABC):
    def __init__(self, variable_ids: set[int] | None, tensor, is_parameter_free: bool,
                 param_size_plus_one: int, id_to_col: dict[int, int], param_to_size: dict[int, int],
                 param_to_col: dict[int, int], var_length):
        self.variable_ids = variable_ids if variable_ids is not None else None
        self._variable_tensor = None if self.is_b(variable_ids) else tensor
        self.constant_data = tensor if self.is_b(variable_ids) else None
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
        self._variable_tensor = self.combine_potentially_none(self._variable_tensor,
                                                              other._variable_tensor)
        self.constant_data = self.combine_potentially_none(self.constant_data, other.constant_data)
        self.is_parameter_free = self.is_parameter_free and other.is_parameter_free
        return self

    @staticmethod
    @abstractmethod
    def combine_potentially_none(a, b):
        pass

    @classmethod
    def get_empty_view(cls, param_size_plus_one, id_to_col, param_to_size,
                       param_to_col: dict[int, int], var_length: int) -> TensorView:
        return cls(None, None, True, param_size_plus_one, id_to_col, param_to_size, param_to_col,
                   var_length)

    @staticmethod
    def is_b(variable_ids):
        return variable_ids == {Constant.ID.value}

    @property
    @abstractmethod
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        """

    @abstractmethod
    def get_A_b(self):
        """
        Returns [A b].
        """
        pass  # noqa

    @abstractmethod
    def select_rows(self, rows: np.ndarray) -> None:
        pass

    @abstractmethod
    def apply_all(self, func: Callable) -> None:
        pass

    @abstractmethod
    def create_new_tensor_view(self, variable_ids, tensor, is_parameter_free: bool):
        pass


class ScipyTensorView(TensorView):

    @property
    def rows(self) -> int:
        if self._variable_tensor is not None:
            return next(iter(next(iter(self._variable_tensor.values())).values()))[0].shape[0]
        elif self.constant_data is not None:
            return next(iter(self.constant_data.values()))[0].shape[0]

        else:
            raise ValueError

    def get_A_b(self):
        return TensorRepresentation.combine([i for i in [self.get_A(), self.get_b()]
                                             if i is not None])

    def select_rows(self, rows: np.ndarray) -> None:

        def func(x):
            return x[rows, :]

        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        if self._variable_tensor is not None:
            self._variable_tensor = {var_id: {k: [func(v_i).tocsr() for v_i in v]
                                              for k, v in parameter_repr.items()}
                                     for var_id, parameter_repr in self._variable_tensor.items()}
        if self.constant_data is not None:
            self.constant_data = {k: [func(v_i).tocsr() for v_i in v]
                                  for k, v in self.constant_data.items()}

    def create_new_tensor_view(self, variable_ids, tensor, is_parameter_free: bool):
        return ScipyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one,
                               self.id_to_col, self.param_to_size, self.param_to_col,
                               self.var_length)

    def accumulate_over_variables(self, func, is_param_free_function: bool):
        if self._variable_tensor is not None:
            for variable_id, tensor in self._variable_tensor.items():
                self._variable_tensor[variable_id] = self.apply_to_parameters(func, tensor) if \
                    is_param_free_function else func(tensor[Constant.ID.value])
        if self.constant_data is not None:
            self.constant_data = self.apply_to_parameters(func, self.constant_data) if \
                is_param_free_function else func(self.constant_data[Constant.ID.value])
        self.is_parameter_free = self.is_parameter_free and is_param_free_function
        return self

    @staticmethod
    def apply_to_parameters(func, parameter_representation):
        return {k: [func(v_i).tocsr() for v_i in v] for k, v in parameter_representation.items()}

    def get_A(self):
        if self._variable_tensor is not None:
            tensor_representations = []
            for variable_id, variable_tensor in self._variable_tensor.items():
                for parameter_id, parameter_tensor in variable_tensor.items():
                    for offset, matrix in enumerate(parameter_tensor):
                        coo_repr = matrix.tocoo(copy=False)
                        tensor_representations.append(TensorRepresentation(
                            np.ones(coo_repr.nnz) * self.param_to_col[parameter_id] + offset,
                            coo_repr.row,
                            coo_repr.col + self.id_to_col[variable_id],
                            coo_repr.data
                        ))
            return TensorRepresentation.combine(tensor_representations)
        else:
            assert self.constant_data is not None
            return None

    def get_b(self):
        if self.constant_data is not None:
            tensor_representations = []
            for parameter_id, parameter_tensor in self.constant_data.items():
                for offset, matrix in enumerate(parameter_tensor):
                    coo_repr = matrix.tocoo(copy=False)
                    tensor_representations.append(TensorRepresentation(
                        np.ones(coo_repr.nnz) * self.param_to_col[parameter_id] + offset,
                        coo_repr.row,
                        np.zeros(coo_repr.nnz) + self.var_length,
                        coo_repr.data
                    ))
            return TensorRepresentation.combine(tensor_representations)
        else:
            assert self._variable_tensor is not None
            return None

    @staticmethod
    def combine_potentially_none(a, b):
        if a is None and b is None:
            return None
        elif a is not None and b is None:
            return a
        elif a is None and b is not None:
            return b
        else:
            return ScipyTensorView.add_dicts(a, b)

    @staticmethod
    def add_dicts(a, b):
        res = {}
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        intersect = keys_a & keys_b
        for key in intersect:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                res[key] = ScipyTensorView.add_dicts(a[key], b[key])
            elif isinstance(a[key], list) and isinstance(b[key], list):
                assert len(a[key]) == len(b[key])
                res[key] = [a + b for a, b in zip(a[key], b[key])]
            else:
                raise ValueError('Values must either be dicts or lists.')
        for key in keys_a - intersect:
            res[key] = a[key]
        for key in keys_b - intersect:
            res[key] = b[key]
        return res