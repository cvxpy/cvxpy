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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.constraints.exponential import ExpCone
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

    def sign_from_args(self):
        """Always unknown.
        """
        return u.Sign.UNKNOWN

    def func_curvature(self):
        """Default curvature.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.INCREASING]

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
