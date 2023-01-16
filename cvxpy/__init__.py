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
from cvxpy.version import (
    version as __version__,)  # cvxpy/version.py is auto-generated
import cvxpy.interface.scipy_wrapper
from cvxpy.atoms import *
from cvxpy.constraints import (Constraint, PSD, SOC, NonPos, NonNeg, Zero, 
                               PowCone3D, PowConeND, ExpCone, 
                               OpRelEntrConeQuad, RelEntrConeQuad, FiniteSet,)
from cvxpy.error import (DCPError, DGPError, DPPError, SolverError,
                         disable_warnings, enable_warnings, warnings_enabled,)
from cvxpy.expressions.constants import CallbackParam, Constant, Parameter
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from cvxpy.transforms import linearize, partial_optimize, suppfunc
from cvxpy.reductions import *
from cvxpy.reductions.solvers.defines import installed_solvers
from cvxpy.settings import (CBC, CLARABEL, COPT, CPLEX, CPP_CANON_BACKEND,
                            CVXOPT, DIFFCP, ECOS, ECOS_BB, GLOP, GLPK, GLPK_MI,
                            GUROBI, INFEASIBLE, INFEASIBLE_INACCURATE, MOSEK,
                            NAG, OPTIMAL, OPTIMAL_INACCURATE, OSQP, PDLP,
                            PROXQP, ROBUST_KKTSOLVER, RUST_CANON_BACKEND, SCIP,
                            SCIPY, SCIPY_CANON_BACKEND, SCS, SDPA,
                            SOLVER_ERROR, UNBOUNDED, UNBOUNDED_INACCURATE,
                            USER_LIMIT, XPRESS, get_num_threads,
                            set_num_threads,)
