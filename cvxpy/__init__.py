"""
Copyright 2017 Steven Diamond

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

__version__ = "0.4.8"
from cvxpy.atoms import *
from cvxpy.expressions.variables import (Variable, Semidef, Symmetric, Bool,
                                         Int, NonNegative)
from cvxpy.expressions.constants import Parameter, CallbackParam, Constant
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.solvers.utilities import installed_solvers
from cvxpy.error import SolverError
from cvxpy.settings import (CVXOPT, GLPK, GLPK_MI, CBC, JULIA_OPT,
                            ECOS, ECOS_BB, SCS, GUROBI, ELEMENTAL, MOSEK, LS,
                            OPTIMAL, UNBOUNDED, INFEASIBLE, SOLVER_ERROR, ROBUST_KKTSOLVER,
                            OPTIMAL_INACCURATE, UNBOUNDED_INACCURATE, INFEASIBLE_INACCURATE)
from cvxpy.transforms import linearize, partial_optimize

# Legacy names.
from cvxpy.expressions.variables.semidef_var import Semidef as semidefinite
