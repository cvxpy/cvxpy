"""
Copyright 2013 Steven Diamond, Eric Chu

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

from elementwise import Elementwise
from cvxpy.expressions.variables import Variable
from cvxpy.constraints.exponential import ExpCone
import cvxpy.utilities as u
import cvxpy.interface as intf
import numpy as np
from scipy.special import xlogy

class entr(Elementwise):
    """Elementwise :math:`-x\log x`.
    """
    def __init__(self, x):
        super(entr, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        results = -xlogy(x, x)
        # Return -inf outside the domain
        results[np.isnan(results)] = -np.inf
        return results

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.NONMONOTONIC]

    def graph_implementation(self, arg_objs):
        rows, cols = self.size
        t = Variable(rows, cols)
        constraints = []
        for i in xrange(rows):
            for j in xrange(cols):
                xi = arg_objs[0][i, j]
                x, y, z = Variable(), Variable(), Variable()
                constraints += [ExpCone(x, y, z),
                                x == t[i, j], y == xi, z == 1]
        return (t, constraints)
