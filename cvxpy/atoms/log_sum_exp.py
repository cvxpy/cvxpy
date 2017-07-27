"""
Copyright 2017 Steven Diamond

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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.elementwise.exp import exp
import numpy as np
from scipy.misc import logsumexp


class log_sum_exp(AxisAtom):
    """:math:`\log\sum_i e^{x_i}`

    """

    def __init__(self, x, axis=None):
        super(log_sum_exp, self).__init__(x, axis=axis)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, sums, and takes the log.
        """
        return logsumexp(values[0], axis=self.axis, keepdims=True)

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
        return (True, False)

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
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        x = arg_objs[0]
        axis = data[0]
        t = lu.create_var(size)

        # sum(exp(x - t)) <= 1
        if axis is None:
            prom_t = lu.promote(t, x.size)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.size)
            obj = lu.sum_entries(obj)

        elif axis == 0:
            prom_size = (x.size[0], 1)
            ones = lu.create_const(np.ones(prom_size), prom_size)
            prom_t = lu.mul_expr(ones, t, x.size)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.size)

            const_size = (1, x.size[0])
            ones = lu.create_const(np.ones(const_size), const_size)
            obj = lu.mul_expr(ones, obj, size)

        else:  # axis == 1
            prom_size = (1, x.size[1])
            ones = lu.create_const(np.ones(prom_size), prom_size)
            prom_t = lu.rmul_expr(t, ones, x.size)
            expr = lu.sub_expr(x, prom_t)
            obj, constraints = exp.graph_implementation([expr], x.size)

            const_size = (x.size[1], 1)
            ones = lu.create_const(np.ones(const_size), const_size)
            obj = lu.rmul_expr(obj, ones, size)

        ones = lu.create_const(np.ones(size), size)
        constraints += [lu.create_leq(obj, ones)]

        return (t, constraints)
