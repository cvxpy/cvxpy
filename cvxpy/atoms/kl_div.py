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
from cvxpy.constraints.exponential import ExpCone
import numpy as np
from scipy.special import xlogy

class kl_div(Atom):
    """:math:`x\log(x/y) - x + y`

    """
    def __init__(self, x, y):
        super(kl_div, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        #TODO return inf outside the domain
        return xlogy(x, x/y) - x + y

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
        """Neither increasing nor decreasing.
        """
        return len(self.args)*[u.monotonicity.NONMONOTONIC]

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        if not self.args[0].is_scalar() or not self.args[1].is_scalar():
            raise ValueError("The arguments to kl_div must resolve to scalars.")

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
        y = arg_objs[1]
        t = lu.create_var((1, 1))
        constraints = [ExpCone(t, x, y),
                       lu.create_geq(y)] # 0 <= y
        # -t - x + y
        obj = lu.sub_expr(y, lu.sum_expr([x, t]))
        return (obj, constraints)
