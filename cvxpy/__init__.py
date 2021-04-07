"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__version__ = "1.1.12"
from cvxpy.atoms import *
from cvxpy.constraints import NonPos, Zero, SOC, PSD
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Parameter, CallbackParam, Constant
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
import cvxpy.interface.scipy_wrapper
from cvxpy.error import DCPError, DPPError, DGPError, SolverError, disable_warnings, enable_warnings, warnings_enabled
from cvxpy.settings import (CVXOPT, GLPK, GLPK_MI, CBC, CPLEX, OSQP, NAG,
                            ECOS, SCS, DIFFCP, GUROBI, MOSEK, XPRESS, SCIP, ECOS_BB,
                            OPTIMAL, UNBOUNDED, INFEASIBLE, SOLVER_ERROR, ROBUST_KKTSOLVER,
                            OPTIMAL_INACCURATE, UNBOUNDED_INACCURATE, INFEASIBLE_INACCURATE, USER_LIMIT)
from cvxpy.settings import get_num_threads, set_num_threads
from cvxpy.transforms import linearize, partial_optimize, suppfunc
from cvxpy.reductions import *
from cvxpy.reductions.solvers.defines import installed_solvers
