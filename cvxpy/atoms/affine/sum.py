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
import builtins
from functools import wraps
from typing import Optional

import numpy as np

import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint


class Sum(AxisAtom, AffAtom):
    """Sum the entries of an expression over a given axis.

    Parameters
    ----------
    expr : Expression
        The expression to sum the entries of.
    axis : None or int or tuple of ints, optional
        The axis or axes along which to sum. The default (axis=None) will
        sum all of the elements of the expression.

        .. versionadded:: 1.6.0

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple.
    keepdims : bool, optional
        If set to true, the axes which are summed over remain in the output
        as dimensions with size one.

    Examples
    --------
    >>> import cvxpy as cp
    >>> x = cp.Variable((2, 3, 4))
    >>> expr = cp.sum(x)
    >>> expr.shape
    ()
    >>> expr = cp.sum(x, axis=0)
    >>> expr.shape
    (3, 4)
    >>> expr = cp.sum(x, axis=(1, 2))
    >>> expr.shape
    (2,)
    """

    def __init__(self, expr, axis=None, keepdims=False) -> None:
        super(Sum, self).__init__(expr, axis=axis, keepdims=keepdims)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?"""
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?"""
        return False

    def numeric(self, values):
        """Sums the entries of value."""
        if intf.is_sparse(values[0]):
            result = np.asarray(values[0].sum(axis=self.axis))
            if not self.keepdims and self.axis is not None:
                result = result.flatten()
        else:
            result = np.sum(values[0], axis=self.axis, keepdims=self.keepdims)
        return result

    def graph_implementation(self,
                            arg_objs: list[lo.LinOp],
                            shape: tuple[int, ...],
                            data=None) -> tuple[lo.LinOp, list[Constraint]]:
        """
        Sum the linear expression's entries.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : int or tuple of ints
            The shape of the resulting expression.
        data : [axis, keepdims] or None
            The axis and keepdims parameters of the sum expression.
        """
        axis, keepdims = data
        # Note: added new case for summing with n-dimensional shapes and 
        # multiple axes. Previous behavior is kept in the else statement.
        if len(arg_objs[0].shape) > 2 or axis not in {None, 0, 1}:
            obj = lu.sum_entries(arg_objs[0], shape=shape, axis=axis, keepdims=keepdims)
        else:
            if axis is None:
                obj = lu.sum_entries(arg_objs[0], shape=shape)
            elif axis == 1:
                if keepdims:
                    const_shape = (arg_objs[0].shape[1], 1)
                else:
                    const_shape = (arg_objs[0].shape[1],)
                ones = lu.create_const(np.ones(const_shape), const_shape)
                obj = lu.rmul_expr(arg_objs[0], ones, shape)
            else:  # axis == 0
                if keepdims:
                    const_shape = (1, arg_objs[0].shape[0])
                else:
                    const_shape = (arg_objs[0].shape[0],)
                ones = lu.create_const(np.ones(const_shape), const_shape)
                obj = lu.mul_expr(ones, arg_objs[0], shape)
        return (obj, [])


@wraps(Sum)
def sum(expr, axis: Optional[int] = None, keepdims: bool = False):
    """
    Wrapper for Sum class.
    """
    if isinstance(expr, list):
        return builtins.sum(expr)
    else:
        return Sum(expr, axis, keepdims)
