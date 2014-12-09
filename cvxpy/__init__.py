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

__version__ = "0.2.15"
from atoms import *
from expressions.variables import Variable, Semidef, BoolVar, IntVar
from expressions.constants import Parameter, CallbackParam, Constant
from problems.problem import Problem
from problems.objective import Maximize, Minimize
import interface.numpy_wrapper
from error import SolverError
from settings import (CVXOPT, ECOS, ECOS_BB, SCS, SCS_MAT_FREE,
OPTIMAL, UNBOUNDED, INFEASIBLE, SOLVER_ERROR,
OPTIMAL_INACCURATE, UNBOUNDED_INACCURATE, INFEASIBLE_INACCURATE)

# Legacy names.
from expressions.variables.semidefinite import Semidef as semidefinite
