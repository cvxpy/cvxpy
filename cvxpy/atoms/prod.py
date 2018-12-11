"""
Copyright 2018 Akshay Agrawal

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

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.interface as intf
import numpy as np


class Prod(AxisAtom):
    """Multiply the entries of an expression.

    The semantics of this atom are the same as np.prod.

    This atom is log-log affine, but it is neither convex nor concave.

    Parameters
    ----------
    expr : Expression
        The expression to multiply the entries of.
    axis : int
        The axis along which to sum.
    keepdims : bool
        Whether to drop dimensions after summing.
    """

    def __init__(self, expr, axis=None, keepdims=False):
        super(Prod, self).__init__(expr, axis=axis, keepdims=keepdims)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.args[0].is_nonneg():
            return (True, False)
        return (False, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def numeric(self, values):
        """Takes the product of the entries of value.
        """
        if intf.is_sparse(values[0]):
            result = np.prod(values[0], axis=self.axis)
            if not self.keepdims and self.axis is not None:
                result = result.A.flatten()
        else:
            result = np.prod(values[0], axis=self.axis, keepdims=self.keepdims)
        return result

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        return np.prod(value) / value

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)


def prod(expr, axis=None, keepdims=False):
    """Multiply the entries of an expression.

    The semantics of this atom are the same as np.prod.

    This atom is log-log affine, but it is neither convex nor concave.

    Parameters
    ----------
    expr : Expression or list[Expression, Numeric]
        The expression to multiply the entries of, or a list of Expressions
        and numeric types.
    axis : int
        The axis along which to take the product; ignored if `expr` is a list.
    keepdims : bool
        Whether to drop dimensions after taking the product; ignored if `expr`
        is a list.
    """
    if isinstance(expr, list):
        return Prod(hstack(expr))
    else:
        return Prod(expr, axis, keepdims)
