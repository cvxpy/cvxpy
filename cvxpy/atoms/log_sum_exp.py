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
from elementwise.exp import exp
from affine.sum import sum as _sum
from .. import utilities as u
from ..expressions.variables import Variable
import numpy as np

class log_sum_exp(Atom):
    """ log(sum(e^x)) """
    def __init__(self, x):
        super(log_sum_exp, self).__init__(x)

    # Evaluates e^x elementwise, sums, and takes the log.
    @Atom.numpy_numeric
    def numeric(self, values):
        exp_mat = np.exp(values[0])
        exp_sum = exp_mat.sum(axis = 1).sum(axis = 0)
        return np.log(exp_sum)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1, 1)

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.INCREASING]

    def graph_implementation(self, arg_objs):
        x = arg_objs[0]
        t = Variable()
        obj, constr = _sum(exp(x - t)).canonical_form
        return (t, constr + [obj <= 1])
