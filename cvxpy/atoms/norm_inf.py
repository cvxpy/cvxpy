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
import numpy as np
from numpy import linalg as LA

class normInf(Atom):
    """Infinity norm; :math:`\max_i\{|x_i|, \dots, |x_n|\}`.
    
    """
    def __init__(self, x):
        super(normInf, self).__init__(x)

    # Returns the Infinity norm of x.
    @Atom.numpy_numeric
    def numeric(self, values):
        cols = values[0].shape[1]
        return max([LA.norm(values[0][:,i], np.inf) for i in range(cols)])

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
        return (t, [-t <= x, x <= t])
