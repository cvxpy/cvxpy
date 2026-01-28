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

Base classes for canonicalization backends.

Backend Architecture Overview
=============================

Tensor Representation
---------------------
The backends represent linear operators as sparse 3D tensors:
- Axis 0 (rows): Corresponds to constraint rows
- Axis 1 (cols): Corresponds to variable columns
- Axis 2 (param): Parameter slices (param_size slices, or 1 if non-parametric)

Key Terms
---------
- param_size: Number of parameter values (e.g., 12 for a 3x4 Parameter)
- param_slice: One 2D slice of the 3D tensor for a specific parameter value
- param_idx: Index identifying which parameter value (0 to param_size-1)
- is_param_free: True if expression has no Parameters (param_size == 1)
- batch_size: Product of batch dimensions in ND arrays (e.g., B1*B2 for shape (B1,B2,m,n))

Class Hierarchy
---------------
CanonBackend (abstract)
  └── PythonCanonBackend (abstract, defines all linop methods)
        ├── SciPyCanonBackend (stacked sparse matrices)
        └── CooCanonBackend (3D COO tensor)

TensorView (abstract)
  └── DictTensorView (abstract)
        ├── SciPyTensorView
        └── CooTensorView

This module contains the abstract base classes used by all backends:
- Constant: Enum for constant ID marker
- TensorRepresentation: Sparse 3D tensor representation
- CanonBackend: Abstract base for all backends
- TensorView: Abstract tensor view
- DictTensorView: Dict-based tensor view
- PythonCanonBackend: Python implementation base class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.lin_ops import LinOp


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


def get_nd_rmul_dims(
    var_shape: Tuple[int, ...],
    const_shape: Tuple[int, ...],
) -> Tuple[int, int, int, bool]:
    """
    Compute dimensions for ND rmul X @ C.

    Parameters
    ----------
    var_shape : tuple
        Shape of the variable X
    const_shape : tuple
        Shape of the constant C

    Returns
    -------
    batch_size : int
        Product of batch dimensions from X (1 if X is 2D)
    m : int
        Second-to-last dimension of X (rows of X, or 1 if 1D row vector)
    n : int
        Last dimension of C (columns of C, or 1 if 1D column vector)
    const_has_batch : bool
        Whether C has batch dimensions (len > 2)
    """
    batch_size = int(np.prod(var_shape[:-2])) if len(var_shape) > 2 else 1
    # 1D variable is a row vector (1, k), so m=1
    m = var_shape[-2] if len(var_shape) >= 2 else 1
    # 1D constant is a column vector (k, 1), so n=1
    n = const_shape[-1] if len(const_shape) >= 2 else 1
    const_has_batch = len(const_shape) > 2
    return batch_size, m, n, const_has_batch


def is_batch_varying(const_shape: Tuple[int, ...]) -> bool:
    """
    Check if constant has batch dimensions with product > 1.

    A batch-varying constant has different values for each batch element,
    requiring the interleaved matrix structure for ND matmul.

    Parameters
    ----------
    const_shape : tuple
        Shape of the constant

    Returns
    -------
    bool
        True if the constant has batch dimensions (shape like (B, m, k) with B > 1)
    """
    if len(const_shape) <= 2:
        return False
    return int(np.prod(const_shape[:-2])) > 1


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
    shape: tuple[int, int]  # (rows, cols)

    def __post_init__(self):
        assert self.data.shape == self.row.shape == self.col.shape == self.parameter_offset.shape

    @classmethod
    def combine(cls, tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Concatenates the row, col, parameter_offset, and data fields of a list of
        TensorRepresentations.
        """
        if not tensors:
            raise ValueError("Cannot combine empty list of tensors")
        # Collect arrays in lists then concatenate once (much faster than repeated np.append)
        data = np.concatenate([t.data for t in tensors])
        row = np.concatenate([t.row for t in tensors])
        col = np.concatenate([t.col for t in tensors])
        parameter_offset = np.concatenate([t.parameter_offset for t in tensors])
        assert all(t.shape == tensors[0].shape for t in tensors)
        return cls(data, row, col, parameter_offset, tensors[0].shape)

    def __eq__(self, other: TensorRepresentation) -> bool:
        return isinstance(other, TensorRepresentation) and \
            np.all(self.data == other.data) and \
            np.all(self.row == other.row) and \
            np.all(self.col == other.col) and \
            np.all(self.parameter_offset == other.parameter_offset) and \
            self.shape == other.shape

    def __add__(self, other: TensorRepresentation) -> TensorRepresentation:
        if self.shape != other.shape:
            raise ValueError("Shapes must match for addition.")
        return TensorRepresentation(
            np.concatenate([self.data, other.data]),
            np.concatenate([self.row, other.row]),
            np.concatenate([self.col, other.col]),
            np.concatenate([self.parameter_offset, other.parameter_offset]),
            self.shape
        )

    @classmethod
    def empty_with_shape(cls, shape: tuple[int, int]) -> TensorRepresentation:
        return cls(
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            shape,
        )

    def flatten_tensor(
        self, num_param_slices: int, order: str = 'F'
    ) -> sp.csc_array:
        """
        Flatten into 2D scipy sparse matrix in order-order and transpose.

        Parameters
        ----------
        num_param_slices: Number of parameter slices.
        order: Whether to order output as 'F' for column-major (default) or 'C' for row-major

        Returns
        -------
        2D sparse array in the requested format.
        """
        if order == 'F':
            rows = (self.col.astype(np.int64) * np.int64(self.shape[0]) + self.row.astype(np.int64))
            cols = self.parameter_offset.astype(np.int64)
        elif order == 'C':
            rows = (self.col.astype(np.int64) + self.row.astype(np.int64) * np.int64(self.shape[1]))
            cols = self.parameter_offset.astype(np.int64)
        else:
            raise ValueError(f"order must be 'F' or 'C', got '{order}'")

        shape = (np.prod(self.shape, dtype=np.int64), num_param_slices)

        return sp.csc_array((self.data, (rows, cols)), shape=shape)

    def get_param_slice(self, param_offset: int) -> sp.csc_array:
        """
        Returns a single slice of the tensor for a given parameter offset.
        """
        mask = self.parameter_offset == param_offset
        return sp.csc_array((self.data[mask], (self.row[mask], self.col[mask])), self.shape)


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

    @abstractmethod
    def build_matrix(
        self, lin_ops: list[LinOp], order: str = 'F'
    ) -> sp.csc_array:
        """
        Main function called from canonInterface.
        Given a list of LinOp trees, each representing a constraint (or the objective), get the
        [A b] Tensor for each, stack them and return the result reshaped as a 2D sparse array
        of shape (total_rows * (var_length + 1)), param_size_plus_one)

        Parameters
        ----------
        lin_ops: list of linOp trees.
        order: Whether [A b] is written in column-major (default) or row-major order

        Returns
        -------
        2D sparse array representing the constraints (or the objective).
        """
        pass  # noqa


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
        self.variable_ids = variable_ids
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
    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
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

    def accumulate_over_variables(self, func: Callable, is_param_free_function: bool) -> TensorView:
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
        """
        Adds the tensor a to b if they are both not none.
        If a (b) is not None but b (a) is None, returns a (b).
        Returns None if both a and b are None.
        """
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
        """
        Returns element-wise addition of two tensors of the same type.
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def tensor_type():
        """
        Returns the type of the underlying tensor
        """
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
                res[key] = self.add_tensors(a[key], b[key])
            else:
                raise ValueError(f'Values must either be dicts or {self.tensor_type()}.')
        for key in keys_a - intersect:
            res[key] = a[key]
        for key in keys_b - intersect:
            res[key] = b[key]
        return res

    @abstractmethod
    def apply_to_parameters(self, func: Callable,
                            parameter_representation: dict[int, Any]) -> dict[int, Any]:
        """
        Apply 'func' to each slice of the parameter representation.
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

    def build_matrix(
        self, lin_ops: list[LinOp], order: str = 'F'
    ) -> sp.csc_array | sp.csr_array:
        self.id_to_col[-1] = self.var_length

        constraint_res = []
        total_rows = sum(np.prod(lin_op.shape) for lin_op in lin_ops)
        row_offset = 0
        for lin_op in lin_ops:
            lin_op_rows = np.prod(lin_op.shape)
            empty_view = self.get_empty_view()
            lin_op_tensor = self.process_constraint(lin_op, empty_view)
            constraint_res.append(lin_op_tensor.get_tensor_representation(row_offset, total_rows))
            row_offset += lin_op_rows
        tensor_res = self.concatenate_tensors(constraint_res)

        self.id_to_col.pop(-1)
        return tensor_res.flatten_tensor(self.param_size_plus_one, order=order)

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
            assert s.ALLOW_ND_EXPR or len(lin_op.shape) in {0, 1, 2}
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
            if lin_op.type in {"concatenate", "vstack", "hstack"}:
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

    def get_constant_data(
        self, lin_op: LinOp, view: TensorView, target_shape: tuple[int, ...] | None
    ) -> tuple[np.ndarray | sp.spmatrix, bool]:
        """
        Extract constant data from a LinOp node.

        Parameters
        ----------
        lin_op : LinOp
            A LinOp node, typically "*_const" or "param", but handles arbitrary types.
        view : TensorView
            Current tensor view (needed for processing parametric expressions).
        target_shape : tuple[int, ...] | None
            Shape to reshape the constant data to.
            - None: Keep column format (m*n, 1). Use for operations that work
              on flattened data (mul_elem, div, kron_r, kron_l).
            - tuple: Reshape to specified shape. Use for matrix operations
              (mul, rmul, conv) that need explicit matrix dimensions.

        Returns
        -------
        data : np.ndarray | sp.spmatrix | dict
            For constants: numpy array or sparse matrix.
            For parametric: dict mapping param_id -> tensor.
        is_param_free : bool
            True if no Parameters, False if parametric.
        """
        # Fast path for constant data to prevent reshape into column vector.
        constants = {"scalar_const", "dense_const", "sparse_const"}
        if target_shape is not None and lin_op.type in constants and lin_op.shape == target_shape:
            constant_data = self.get_constant_data_from_const(lin_op)
            return constant_data, True

        constant_view = self.process_constraint(lin_op, view)
        assert constant_view.variable_ids == {Constant.ID.value}
        constant_data = constant_view.tensor[Constant.ID.value]
        if target_shape is not None:
            # Reshape from column format to the requested shape.
            constant_data = self.reshape_constant_data(constant_data, target_shape)

        data_to_return = constant_data[Constant.ID.value] if constant_view.is_parameter_free \
            else constant_data
        return data_to_return, constant_view.is_parameter_free

    @staticmethod
    @abstractmethod
    def get_constant_data_from_const(lin_op: LinOp) -> np.ndarray | sp.spmatrix:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        pass  # noqa

    @staticmethod
    @abstractmethod
    def reshape_constant_data(constant_data: Any, target_shape: tuple[int, ...]) -> Any:
        """
        Reshape constant data from column format to the target shape.
        """
        pass  # noqa

    @staticmethod
    def concatenate_tensors(tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Takes list of tensors which have already been offset along axis 0 (rows) and
        combines them into a single tensor.
        """
        return TensorRepresentation.combine(tensors)

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
            "broadcast_to": self.broadcast_to,
            "neg": self.neg,
            "mul_elem": self.mul_elem,
            "sum_entries": self.sum_entries,
            "div": self.div,
            "reshape": self.reshape,
            "index": self.index,
            "diag_vec": self.diag_vec,
            "hstack": self.hstack,
            "vstack": self.vstack,
            "concatenate": self.concatenate,
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
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the rhs.
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.
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
    @abstractmethod
    def broadcast_to(lin: LinOp, view: TensorView) -> TensorView:
        """
        Broadcast view to a new shape.
        """
        pass  # noqa

    @staticmethod
    def neg(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, return (-A, -b).
        """

        def func(x, _p):
            return -x

        view.apply_all(func)
        return view

    @abstractmethod
    def mul_elem(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        d is broadcasted along dimension 1 (columns).
        When the lhs is parametrized, multiply elementwise each slice of the tensor with the
        single, constant slice of the rhs.
        Otherwise, multiply elementwise the single slice of the tensor with each slice of the rhs.
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
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data instead.

        Note: div currently doesn't support parameters.
        """
        pass  # noqa

    @staticmethod
    def index(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, select the rows corresponding to the elements of the expression being
        indexed. Supports an arbitrary number of dimensions.
        """
        indices = [np.arange(s.start, s.stop, s.step) for s in lin.data]
        assert len(indices) > 0
        rows = indices[0]
        cum_prod = np.cumprod([lin.args[0].shape])
        for i in range(1, len(indices)):
            product_size = cum_prod[i - 1]
            # add new indices to rows and apply offset to all previous indices
            offset = np.add.outer(rows, indices[i] * product_size).flatten(order="F")
            rows = offset
        view.select_rows(rows)
        return view

    @staticmethod
    @abstractmethod
    def diag_vec(lin: LinOp, view: TensorView) -> TensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
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
        total_rows = sum(np.prod(arg.shape, dtype=int) for arg in lin.args)
        res = None
        for arg in lin.args:
            arg_view = self.process_constraint(arg, view)
            func = self.get_stack_func(total_rows, offset)
            arg_view.apply_all(func)
            arg_rows = np.prod(arg.shape, dtype=int)
            offset += arg_rows
            if res is None:
                res = arg_view
            else:
                res += arg_view
        assert res is not None
        return res

    def concatenate(self, lin: LinOp, view: TensorView) -> TensorView:
        """Concatenate multiple tensors along a specified axis.

        This method performs the concatenation of multiple tensors, following NumPy's behavior.
        It correctly maps the indices from the input tensors to the concatenated output tensor,
        ensuring that elements are placed in the correct positions in the resulting tensor.
        """
        res = self.hstack(lin=lin, view=view)
        axis = lin.data[0]
        if axis is None:
            # In this case following numpy, arrays are flattened in 'C' order
            order = np.arange(sum(np.prod(arg.shape, dtype=int) for arg in lin.args))
            res.select_rows(order)
            return res

        offset = 0
        indices = []
        for arg in lin.args:
            arg_rows = np.prod(arg.shape, dtype=int)
            indices.append(np.arange(arg_rows).reshape(arg.shape, order="F") + offset)
            offset += arg_rows
        order = np.concatenate(indices, axis=axis).flatten(order="F").astype(int)
        res.select_rows(order)
        return res

    def vstack(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given views (A0,b0), (A1,b1),..., (An,bn), first, stack them along axis 0 via hstack.
        Then, permute the rows of the resulting tensor to be consistent with stacking the arguments
        vertically instead of horizontally.
        """
        res = self.hstack(lin=lin, view=view)
        offset = 0
        indices = []
        for arg in lin.args:
            arg_rows = np.prod(arg.shape, dtype=int)
            indices.append(np.arange(arg_rows).reshape(arg.shape, order="F") + offset)
            offset += arg_rows
        order = np.vstack(indices).flatten(order="F").astype(int)
        res.select_rows(order)
        return res

    @staticmethod
    def transpose(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, permute the rows such that they correspond to the transposed
        expression with arbitrary axis permutation.
        """
        axes = lin.data[0]
        original_shape = lin.args[0].shape
        indices = np.arange(np.prod(original_shape)).reshape(original_shape, order='F')
        transposed_indices = np.transpose(indices, axes)
        rows = transposed_indices.flatten(order='F')
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
        elements on the diagonal in the original expression.
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
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
        view.select_rows(diag_indices.astype(int))
        return view

    @abstractmethod
    def rmul(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Multiply view with constant data from the right.
        When the rhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the lhs.
        Otherwise, multiply the single slice of the tensor with each slice of the lhs.
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
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        pass  # noqa

    @abstractmethod
    def kron_r(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).

        Note: kron_r currently doesn't support parameters.
        """
        pass  # noqa

    @abstractmethod
    def kron_l(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).

        Note: kron_l currently doesn't support parameters.
        """
        pass  # noqa

    @staticmethod
    def _get_kron_row_indices(lhs_shape, rhs_shape):
        """
        Internal function that computes the row indices corresponding to the
        kronecker product of two sparse tensors.
        """
        rhs_ones = np.ones(rhs_shape)
        lhs_ones = np.ones(lhs_shape)

        rhs_arange = np.arange(np.prod(rhs_shape)).reshape(rhs_shape, order="F")
        lhs_arange = np.arange(np.prod(lhs_shape)).reshape(lhs_shape, order="F")

        row_indices = (np.kron(lhs_ones, rhs_arange) +
                       np.kron(lhs_arange, rhs_ones * np.prod(rhs_shape))) \
            .flatten(order="F").astype(int)
        return row_indices

    @abstractmethod
    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> Any:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
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
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        """
        pass  # noqa
