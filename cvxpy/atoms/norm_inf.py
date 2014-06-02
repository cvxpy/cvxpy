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

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from numpy import linalg as LA

class normInf(Atom):
    """Infinity norm; :math:`\max_i\{|x_i|, \dots, |x_n|\}`.

    """
    def __init__(self, x):
        super(normInf, self).__init__(x)

    # Returns the Infinity norm of x.
    @Atom.numpy_numeric
    def numeric(self, values):
        cols = values[0].shape[1]
        return max([LA.norm(values[0][:,i], np.inf) for i in range(cols)])

    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        return [u.monotonicity.SIGNED]

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
        t = lu.create_var((1, 1))
        promoted_t = lu.promote(t, x.size)
        constraints = [lu.create_geq(lu.sum_expr([x, promoted_t])),
                       lu.create_leq(x, promoted_t)]
        return (t, constraints)
