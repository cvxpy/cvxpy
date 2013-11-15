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

from ... import utilities as u
from ... import interface as intf
from ...expressions import types
from ...expressions.variables import Variable
from elementwise import Elementwise
import numpy as np

class abs(Elementwise):
    """ Elementwise absolute value """
    def __init__(self, x):
        super(abs, self).__init__(x)

    # Returns the elementwise absolute value of x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.absolute(values[0])

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.SIGNED]

    def graph_implementation(self, arg_objs):
        x = arg_objs[0]
        t = Variable(*self.size)
        constraints = [-t <= x, x <= t]
        return (t, constraints)