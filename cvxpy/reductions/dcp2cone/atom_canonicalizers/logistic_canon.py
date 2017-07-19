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

from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import \
    exp_canon
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
import numpy as np


def logistic_canon(expr, args):
    x = args[0]
    shape = expr.shape
    # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
    t0 = Variable(shape)
    t1, constr1 = exp_canon(expr, [-t0])
    t2, constr2 = exp_canon(expr, [x - t0])
    ones = Constant(np.ones(shape))
    constraints = constr1 + constr2 + [t1 + t2 <= ones]
    return t0, constraints
