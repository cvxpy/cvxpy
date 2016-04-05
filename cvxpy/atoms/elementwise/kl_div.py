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
from cvxpy.constraints.exponential import ExpCone
import numpy as np
from scipy.special import xlogy

class kl_div(Elementwise):
    """:math:`x\log(x/y) - x + y`

    """
    def __init__(self, x, y):
        super(kl_div, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        #TODO return inf outside the domain
        return xlogy(x, x/y) - x + y

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Neither increasing nor decreasing.
        """
        return len(self.args)*[u.monotonicity.NONMONOTONIC]

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
        x = Elementwise._promote(arg_objs[0], size)
        y = Elementwise._promote(arg_objs[1], size)
        t = lu.create_var(size)
        constraints = [ExpCone(t, x, y),
                       lu.create_geq(y)] # 0 <= y
        # -t - x + y
        obj = lu.sub_expr(y, lu.sum_expr([x, t]))
        return (obj, constraints)
