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

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.interface as intf
import numpy as np
from functools import wraps


class Sum(AxisAtom, AffAtom):
    """Sum the entries of an expression.

    Parameters
    ----------
    expr : Expression
        The expression to sum the entries of.
    axis : int
        The axis along which to sum.
    keepdims : bool
        Whether to drop dimensions after summing.
    """

    def __init__(self, expr, axis=None, keepdims=False):
        super(Sum, self).__init__(expr, axis=axis, keepdims=keepdims)

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?"""
        return True

    def numeric(self, values):
        """Sums the entries of value.
        """
        if intf.is_sparse(values[0]):
            result = np.sum(values[0], axis=self.axis)
            if not self.keepdims and self.axis is not None:
                result = result.A.flatten()
        else:
            result = np.sum(values[0], axis=self.axis, keepdims=self.keepdims)
        return result

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Sum the linear expression's entries.

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
        axis = data[0]
        keepdims = data[1]
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
def sum(expr, axis=None, keepdims=False):
    """Wrapper for Sum class.
    """
    if isinstance(expr, list):
        return __builtins__['sum'](expr)
    else:
        return Sum(expr, axis, keepdims)
