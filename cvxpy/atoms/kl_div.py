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
from ..constraints import ExpCone
from .. import utilities as u
from ..expressions.variables import Variable
import numpy as np
from scipy.special import xlogy

class kl_div(Atom):
    """:math:`x\log(x/y) - x + y`

    """
    def __init__(self, x, y):
        super(kl_div, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        #TODO return inf outside the domain
        return xlogy(x, x/y) - x + y

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1, 1)

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return len(self.args)*[u.monotonicity.NONMONOTONIC]

    # Only scalar arguments are valid.
    def validate_arguments(self):
        if not self.args[0].is_scalar() or not self.args[1].is_scalar():
            raise TypeError("The arguments to kl_div must resolve to scalars." )

    def graph_implementation(self, arg_objs):
        x = arg_objs[0]
        y = arg_objs[1]
        t = Variable()
        # Duplicate variables for x, y.
        xc, yc = Variable(), Variable()
        constraints = [ExpCone(t, xc, yc), y >= 0,
                       xc == x, yc == y]
        return (-t - x + y, constraints)
