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
from cvxpy.constraints.second_order import SOC
import numpy as np

class quad_over_lin(Atom):
    """ :math:`x^Tx/y`

    """
    def __init__(self, x, y):
        super(quad_over_lin, self).__init__(x, y)

    # Returns the dot product of x divided by y.
    @Atom.numpy_numeric
    def numeric(self, values):
        return np.dot(values[0].T, values[0])/values[1]

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    # Increasing for positive entry of x, decreasing for negative.
    def monotonicity(self):
        return [u.monotonicity.SIGNED, u.monotonicity.DECREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise TypeError("The first argument to quad_over_lin must be a vector.")
        elif not self.args[1].is_scalar():
            raise TypeError("The second argument to quad_over_lin must be a scalar.")

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
        y = arg_objs[1] # Known to be a scalar.
        v = lu.create_var((1, 1))
        two = lu.create_const(2, (1, 1))
        constraints = [SOC(lu.sum_expr([y, v]),
                           [lu.sub_expr(y, v),
                            lu.mul_expr(two, x, x.size)]),
                       lu.create_geq(y)]
        return (v, constraints)
