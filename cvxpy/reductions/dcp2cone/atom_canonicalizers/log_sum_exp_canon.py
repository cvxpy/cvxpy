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

import numpy as np

from cvxpy.atoms import exp
from cvxpy.atoms import promote, reshape
from cvxpy.atoms import sum
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import exp_canon


def log_sum_exp_canon(expr, args):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    keepdims = expr.keepdims
    t = Variable(shape)

    # log(sum(exp(x))) <= t <=> sum(exp(x-t)) <= 1
    if axis is None:  # shape = (1, 1)
        promoted_t = promote(t, x.shape)
    elif axis == 0:  # shape = (1, n)
        promoted_t = Constant(np.ones((x.shape[0], 1))) * reshape(
                                                        t, (1,) + x.shape[1:])
    else:  # shape = (m, 1)
        promoted_t = reshape(t, x.shape[:-1] + (1,)) * Constant(
                                                      np.ones((1, x.shape[1])))

    exp_expr = exp(x - promoted_t)
    obj, constraints = exp_canon(exp_expr, exp_expr.args)
    obj = sum(obj, axis=axis, keepdims=keepdims)
    ones = Constant(np.ones(shape))
    constraints.append(obj <= ones)
    return t, constraints
