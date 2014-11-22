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
from cvxpy.constraints import SOC_Elemwise
import numpy as np

class geo_mean(Elementwise):
    """ Elementwise geometric mean; :math:`(x_1, x_2)^{1/2}`. """
    def __init__(self, x, y):
        super(geo_mean, self).__init__(x, y)

    # Returns the geometric mean of x and y.
    def numeric(self, values):
        return np.sqrt(np.multiply(values[0], values[1]))

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return len(self.args)*[u.monotonicity.INCREASING]

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
        # Promote scalars.
        for idx, arg in enumerate(arg_objs):
            if arg.size != size:
                arg_objs[idx] = lu.promote(arg, size)
        x = arg_objs[0]
        y = arg_objs[1]
        v = lu.create_var(x.size)
        two = lu.create_const(2, (1, 1))
        # SOC(x + y, [y - x, 2*v])
        constraints = [
            SOC_Elemwise(lu.sum_expr([x, y]),
                         [lu.sub_expr(y, x),
                          lu.mul_expr(two, v, v.size)])
        ]
        # 0 <= x, 0 <= y
        constraints += [lu.create_geq(x), lu.create_geq(y)]
        return (v, constraints)
