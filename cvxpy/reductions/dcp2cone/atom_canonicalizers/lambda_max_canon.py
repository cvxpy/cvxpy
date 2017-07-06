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

from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def lambda_max_canon(expr, args):
    A = args[0]
    shape = expr.shape
    t = Variable(*shape)
    # SDP constraint: I*t - A
    expr = Constant(np.eye(A.shape[0])) * t - A
    return t, [PSD(expr)]
