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

from atoms import *
from expressions.variables import Variable, semidefinite
from expressions.constants import Parameter
from expressions.constants import Constant
from problems.problem import Problem
from problems.objective import Maximize, Minimize
import interface.numpy_wrapper
from settings import (CVXOPT, ECOS, SCS,
OPTIMAL, UNBOUNDED, INFEASIBLE, SOLVER_ERROR,
OPTIMAL_INACCURATE, UNBOUNDED_INACCURATE, INFEASIBLE_INACCURATE)

