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

from elementwise import Elementwise
from ..quad_over_lin import quad_over_lin
from ... import utilities as u
from ... import interface as intf
from ...expressions import types
from ...expressions.variables import Variable
import numpy as np

class square(Elementwise):
    """ Elementwise square """
    def __init__(self, x):
        super(square, self).__init__(x)

    # Returns the elementwise square of x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.square(values[0])

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.SIGNED]

    def graph_implementation(self, arg_objs):
        rows, cols = self.size
        t = Variable(rows, cols)
        constraints = []
        for i in xrange(rows):
            for j in xrange(cols):
                xi = arg_objs[0][i,j]
                obj, constr = quad_over_lin(xi, 1).canonical_form
                constraints += constr + [obj <= t[i,j]]
        return (t, constraints)
