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
from ...expressions import types
from ...expressions.variables import Variable
from max import max
import numpy as np

class min(max):
    """ Elementwise minimum. """
    # Returns the elementwise minimum.
    @max.numpy_numeric
    def numeric(self, values):
        return reduce(np.minimum, values)

    """
    Reduces the list of argument signs according to the following rules:
        NEGATIVE, ANYTHING = NEGATIVE
        ZERO, UNKNOWN = NEGATIVE
        ZERO, ZERO = ZERO
        ZERO, POSITIVE = ZERO
        UNKNOWN, POSITIVE = UNKNOWN
        POSITIVE, POSITIVE = POSITIVE
    """
    def sign_from_args(self):
        neg_mat = self.args[0]._dcp_attr.sign.neg_mat
        pos_mat = self.args[0]._dcp_attr.sign.pos_mat
        for arg in self.args[1:]:
            neg_mat = neg_mat | arg._dcp_attr.sign.neg_mat
            pos_mat = pos_mat & arg._dcp_attr.sign.pos_mat
        return u.Sign(neg_mat, pos_mat)

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def graph_implementation(self, arg_objs):
        t = Variable(*self.size)
        constraints = [t <= x for x in arg_objs]
        return (t, constraints)