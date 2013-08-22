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
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.second_order import SOC
import cvxpy.utilities as u

class norm2(Atom):
    """ L2 norm (sum(x^2))^(1/2) """
    def __init__(self, x):
        super(norm2, self).__init__(x)

    def set_shape(self):
        self.validate_arguments()
        self._shape = u.Shape(1,1)

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise TypeError("The argument '%s' to norm2 must resolve to a vector." 
                % self.args[0].name())

    @staticmethod
    def graph_implementation(var_args, size):
        x = var_args[0]
        t = Variable()
        return (t, [SOC(t, x)])