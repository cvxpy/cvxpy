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
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.exp import exp
import numpy as np

class logistic(Elementwise):
    """:math:`\log(1 + e^{x})`

    This is a special case of log(sum(exp)) that is evaluates to a vector rather
    than to a scalar which is useful for logistic regression.
    """
    def __init__(self, x):
        super(logistic, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, adds 1, and takes the log.
        """
        return np.logaddexp(0, values[0])

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

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
        t = lu.create_var(size)

        # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
        obj0, constr0 = exp.graph_implementation([lu.neg_expr(t)], size)
        obj1, constr1 = exp.graph_implementation([lu.sub_expr(x, t)], size)
        lhs = lu.sum_expr([obj0, obj1])
        ones = lu.create_const(np.mat(np.ones(size)), size)
        constr = constr0 + constr1 + [lu.create_leq(lhs, ones)]

        return (t, constr)
