"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.elementwise.exp import exp
import numpy as np
from scipy.misc import logsumexp


class log_sum_exp(AxisAtom):
    """:math:`\log\sum_i e^{x_i}`

    """

    def __init__(self, x, axis=None, keepdims=False):
        super(log_sum_exp, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, sums, and takes the log.
        """
        return logsumexp(values[0], axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        denom = np.exp(logsumexp(value, axis=None, keepdims=True))
        nom = np.exp(value)
        D = nom/denom
        return D

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Reduces the atom to an affine expression and list of constraints.

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
        x = arg_objs[0]
        axis = data[0]
        t = lu.create_var(shape)

        # sum(exp(x - t)) <= 1
        if axis is None:
            prom_t = lu.promote(t, x.shape)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.shape)
            obj = lu.sum(obj)

        elif axis == 0:
            prom_shape = (x.shape[0], 1)
            ones = lu.create_const(np.ones(prom_shape), prom_shape)
            prom_t = lu.mul_expr(ones, t, x.shape)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.shape)

            const_shape = (1, x.shape[0])
            ones = lu.create_const(np.ones(const_shape), const_shape)
            obj = lu.mul_expr(ones, obj, shape)

        else:  # axis == 1
            prom_shape = (1, x.shape[1])
            ones = lu.create_const(np.ones(prom_shape), prom_shape)
            prom_t = lu.rmul_expr(t, ones, x.shape)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.shape)

            const_shape = (x.shape[1], 1)
            ones = lu.create_const(np.ones(const_shape), const_shape)
            obj = lu.rmul_expr(obj, ones, shape)

        ones = lu.create_const(np.ones(shape), shape)
        constraints += [lu.create_leq(obj, ones)]

        return (t, constraints)
