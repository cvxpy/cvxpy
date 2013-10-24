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
from ..constraints.affine import AffEqConstraint, AffLeqConstraint
from ..constraints.second_order import SOC
from affine.vstack import vstack
import math

class geo_mean(Atom):
    """ Geometric mean of two scalars """
    def __init__(self, x, y):
        super(geo_mean, self).__init__(x, y)

    # Returns the geometric mean of x and y.
    def numeric(self, values):
        return math.sqrt(values[0]*values[1])

    # The shape is the common width and the sum of the heights.
    def set_shape(self):
        self.validate_arguments()
        self._shape = u.Shape(1,1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN
        
    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return len(self.args)*[u.Monotonicity.INCREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        if not self.args[0].is_scalar() or not self.args[1].is_scalar():
            raise TypeError("The arguments to geo_mean must resolve to scalars." )
    
    @staticmethod
    def graph_implementation(var_args, size):
        v = Variable(*size)
        x = var_args[0]
        y = var_args[1]
        obj,constraints = vstack.graph_implementation([y - x, v + v],
                                                      (y.size[0] + 1,1))
        constraints += [SOC(x + y, obj),
                        AffLeqConstraint(0, x),
                        AffLeqConstraint(0, y)]
        return (v, constraints)