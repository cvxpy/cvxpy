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

__version__ = "0.3.1"
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, Semidef, Symmetric, Bool, Int
from cvxpy.expressions.constants import Parameter, CallbackParam, Constant
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.solvers.utilities import installed_solvers
from cvxpy.error import SolverError
from cvxpy.settings import (CVXOPT, GLPK, GLPK_MI,
ECOS, ECOS_BB, SCS, GUROBI, ELEMENTAL, MOSEK,
OPTIMAL, UNBOUNDED, INFEASIBLE, SOLVER_ERROR, ROBUST_KKTSOLVER,
OPTIMAL_INACCURATE, UNBOUNDED_INACCURATE, INFEASIBLE_INACCURATE)
from cvxpy.transforms import partial_optimize

# Legacy names.
from cvxpy.expressions.variables.semidef_var import Semidef as semidefinite
