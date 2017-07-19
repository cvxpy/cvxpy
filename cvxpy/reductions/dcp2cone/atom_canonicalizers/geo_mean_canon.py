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

from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs


def geo_mean_canon(expr, args):
    x = args[0]
    w = expr.w
    shape = expr.shape
    t = Variable(shape)

    x_list = [x[i] for i in range(len(w))]

    # todo: catch cases where we have (0, 0, 1)?
    # todo: what about curvature case (should be affine) in trivial
    #       case of (0, 0 , 1)?
    # should this behavior match with what we do in power?
    return t, gm_constrs(t, x_list, w)
