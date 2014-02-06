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
from ..expressions.variables import Variable
from ..constraints.second_order import SOC
from numpy import linalg as LA

class norm2(Atom):
    """L2 norm; :math:`(\sum_i x_i^2)^{1/2}`.
    
    """
    def __init__(self, x):
        super(norm2, self).__init__(x)

    # Returns the L2 norm of x for vector x
    # and the Frobenius norm for matrix x.
    @Atom.numpy_numeric
    def numeric(self, values):
        return LA.norm(values[0])

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

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
        t = Variable()
        return (t, [SOC(t, [x])])
