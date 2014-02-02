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

from atom import Atom
from .. import utilities as u
from .. import interface as intf
from ..expressions import types
from ..expressions.variables import Variable
from ..constraints.second_order import SOC
from affine.vstack import vstack
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

    def graph_implementation(self, arg_objs):
        v = Variable()
        x = arg_objs[0]
        y = arg_objs[1]
        constraints = [SOC(y + v, [y - v, 2*x]), 0 <= y]
        return (v, constraints)
