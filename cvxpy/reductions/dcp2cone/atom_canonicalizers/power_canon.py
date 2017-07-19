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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
import numpy as np


def power_canon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    else:
        t = Variable(shape)
        # TODO(akshayka): gm_constrs requires each of its inputs to be a Variable;
        # is this something that we want to change?
        if 0 < p < 1:
            return t, gm_constrs(t, [x, ones], w)
        elif p > 1:
            return t, gm_constrs(x, [t, ones], w)
        elif p < 0:
            return t, gm_constrs(ones, [x, t], w)
        else:
            raise NotImplementedError('This power is not yet supported.')
