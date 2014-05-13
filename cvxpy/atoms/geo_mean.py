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
from cvxpy.constraints import SOC
import math

class geo_mean(Atom):
    """ Geometric mean of two scalars; :math:`(x_1, \cdots, x_n)^{1/n}`. """
    def __init__(self, x, y):
        super(geo_mean, self).__init__(x, y)

    # Returns the geometric mean of x and y.
    def numeric(self, values):
        return math.sqrt(values[0]*values[1])

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self):
        return u.Shape(1, 1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return len(self.args)*[u.monotonicity.INCREASING]

    # Only scalar arguments are valid.
    def validate_arguments(self):
        if not self.args[0].is_scalar() or not self.args[1].is_scalar():
            raise TypeError("The arguments to geo_mean must resolve to scalars." )

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
        # TODO use log for n != 2.
        v = lu.create_var((1, 1))
        x = arg_objs[0]
        y = arg_objs[1]
        two = lu.create_const(2, (1, 1))
        # SOC(x + y, [y - x, 2*v])
        constraints = [
            SOC(lu.sum_expr([x, y]),
                [lu.sub_expr(y, x),
                 lu.mul_expr(two, v, (1, 1))])
        ]
        # 0 <= x, 0 <= y
        constraints += [lu.create_geq(x), lu.create_geq(y)]
        return (v, constraints)
