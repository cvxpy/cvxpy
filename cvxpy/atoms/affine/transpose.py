"""
Copyright 2013 Steven Diamond

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
from typing import List, Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


class transpose(AffAtom):
    """Transpose an expression.
    
    For an n-D expression, if axes are given, the order indicates the permutation of axes.
    """

    def __init__(self, expr, axes=None) -> None:
        self.axes = axes
        super(AffAtom, self).__init__(expr)

    # The string representation of the atom.
    def name(self) -> str:
        if self.axes is None:
            return f"{self.args[0]}.T"
        else:
            return f"transpose({self.args[0]}, axes={self.axes})"

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.transpose(values[0], axes=self.axes)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_skew_symmetric(self) -> bool:
        """Is the expression skew-symmetric?
        """
        return self.args[0].is_skew_symmetric()

    def is_hermitian(self) -> bool:
        """Is the expression Hermitian?
        """
        return self.args[0].is_hermitian()

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the transpose expression.
        """
        arr = np.empty(self.args[0].shape, dtype=np.dtype([]))
        return np.transpose(arr, axes=self.axes).shape

    def get_data(self):
        """Returns the axes for transposition.
        """
        return [self.axes]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Create a new variable equal to the argument transposed.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.transpose(arg_objs[0], self.axes), [])

def permute_dims(expr, axes: List[int]):
    """Permute the dimensions of the expression.

    Alias for transpose with specified axes.

    Parameters
    ----------
    expr : AffAtom
        The expression to permute dimensions of.
    axes : list or tuple of int
        The new order of the axes.

    Returns
    -------
    AffAtom
        A transpose atom with the specified axes.
    """
    return transpose(expr, axes=axes)

def swapaxes(expr, axis1: int, axis2: int):
    """Swap two axes of the expression.

    Parameters
    ----------
    expr : AffAtom
        The expression to swap axes of.
    axis1 : int
        The first axis to swap.
    axis2 : int
        The second axis to swap.

    Returns
    -------
    AffAtom
        A transpose atom with the axes swapped.
    """
    axes = list(range(len(expr.shape)))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return transpose(expr, axes=axes)

def moveaxis(expr, source: List[int], destination: List[int]):
    """Move axes of the expression to new positions.

    Parameters
    ----------
    expr : AffAtom
        The expression to move axes of.
    source : list of int
        The original positions of the axes to move.
    destination : list of int
        The new positions for the moved axes.

    Returns
    -------
    AffAtom
        A new transpose atom with the axes moved.
    """
    if not isinstance(source, list) or not isinstance(destination, list):
        raise TypeError("Source and destination must be lists of integers.")
    
    if len(source) != len(destination):
        raise ValueError("Source and destination must have the same length.")

    order = [n for n in range(expr.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return transpose(expr, axes=order)
