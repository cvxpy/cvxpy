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
    version as __version__,
)  # cvxpy/version.py is auto-generated
from cvxpy.atoms import *
from cvxpy.constraints import (
    Constraint as Constraint,
    Cone as Cone,
    PSD as PSD,
    SOC as SOC,
    NonPos as NonPos,
    NonNeg as NonNeg,
    Zero as Zero,
    PowCone3D as PowCone3D,
    PowConeND as PowConeND,
    ExpCone as ExpCone,
    OpRelEntrConeQuad as OpRelEntrConeQuad,
    RelEntrConeQuad as RelEntrConeQuad,
    FiniteSet as FiniteSet,
)
from cvxpy.error import (
    DCPError as DCPError,
    DGPError as DGPError,
    DPPError as DPPError,
    SolverError as SolverError,
    disable_warnings as disable_warnings,
    enable_warnings as enable_warnings,
    warnings_enabled as warnings_enabled,
)
from cvxpy.expressions.constants import (
    CallbackParam as CallbackParam,
    Constant as Constant,
    Parameter as Parameter,
)
from cvxpy.expressions.expression import Expression as Expression
from cvxpy.expressions.variable import Variable as Variable
from cvxpy.problems.objective import (
    Maximize as Maximize,
    Minimize as Minimize,
    Objective as Objective,
)
from cvxpy.problems.problem import Problem as Problem
from cvxpy.transforms import (
    linearize as linearize,
    partial_optimize as partial_optimize,
    suppfunc as suppfunc,
)
from cvxpy.reductions.solvers.defines import installed_solvers as installed_solvers
from cvxpy.settings import (
    CBC as CBC,
    CLARABEL as CLARABEL,
    COPT as COPT,
    CPLEX as CPLEX,
    CPP_CANON_BACKEND as CPP_CANON_BACKEND,
    CVXOPT as CVXOPT,
    DIFFCP as DIFFCP,
    ECOS as ECOS,
    ECOS_BB as ECOS_BB,
    GLOP as GLOP,
    GLPK as GLPK,
    GLPK_MI as GLPK_MI,
    GUROBI as GUROBI,
    INFEASIBLE as INFEASIBLE,
    INFEASIBLE_INACCURATE as INFEASIBLE_INACCURATE,
    MOSEK as MOSEK,
    NAG as NAG,
    OPTIMAL as OPTIMAL,
    OPTIMAL_INACCURATE as OPTIMAL_INACCURATE,
    OSQP as OSQP,
    DAQP as DAQP,
    PDLP as PDLP,
    PIQP as PIQP,
    PROXQP as PROXQP,
    ROBUST_KKTSOLVER as ROBUST_KKTSOLVER,
    RUST_CANON_BACKEND as RUST_CANON_BACKEND,
    SCIP as SCIP,
    SCIPY as SCIPY,
    SCIPY_CANON_BACKEND as SCIPY_CANON_BACKEND,
    SCS as SCS,
    SDPA as SDPA,
    SOLVER_ERROR as SOLVER_ERROR,
    UNBOUNDED as UNBOUNDED,
    UNBOUNDED_INACCURATE as UNBOUNDED_INACCURATE,
    USER_LIMIT as USER_LIMIT,
    XPRESS as XPRESS,
    get_num_threads as get_num_threads,
    set_num_threads as set_num_threads,
)
