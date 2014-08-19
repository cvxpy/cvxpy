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
from cvxpy.atoms.elementwise.square import square
import numpy as np

class sqrt(Elementwise):
    """ Elementwise square root """
    def __init__(self, x):
        super(sqrt, self).__init__(x)

    # Returns the elementwise square root of x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.sqrt(values[0])

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

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
        # x >= 0 implied by x >= t^2.
        obj, constraints = square.graph_implementation([t], size)
        return (t, constraints + [lu.create_leq(obj, x), lu.create_geq(t)])
