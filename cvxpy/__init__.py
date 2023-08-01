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
import cvxpy.atoms
import cvxpy.interface.scipy_wrapper as scipy_wrapper
from cvxpy.atoms import *
from cvxpy.constraints import (
    PSD,
    SOC,
    Constraint,
    ExpCone,
    FiniteSet,
    NonNeg,
    NonPos,
    OpRelEntrConeQuad,
    PowCone3D,
    PowConeND,
    RelEntrConeQuad,
    Zero,
)
from cvxpy.error import (
    DCPError,
    DGPError,
    DPPError,
    SolverError,
    disable_warnings,
    enable_warnings,
    warnings_enabled,
)
from cvxpy.expressions.constants import CallbackParam, Constant, Parameter
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize, Objective
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.defines import installed_solvers
from cvxpy.settings import (
    CBC,
    CLARABEL,
    COPT,
    CPLEX,
    CPP_CANON_BACKEND,
    CVXOPT,
    DIFFCP,
    ECOS,
    ECOS_BB,
    GLOP,
    GLPK,
    GLPK_MI,
    GUROBI,
    INFEASIBLE,
    INFEASIBLE_INACCURATE,
    MOSEK,
    NAG,
    OPTIMAL,
    OPTIMAL_INACCURATE,
    OSQP,
    PDLP,
    PIQP,
    PROXQP,
    ROBUST_KKTSOLVER,
    RUST_CANON_BACKEND,
    SCIP,
    SCIPY,
    SCIPY_CANON_BACKEND,
    SCS,
    SDPA,
    SOLVER_ERROR,
    UNBOUNDED,
    UNBOUNDED_INACCURATE,
    USER_LIMIT,
    XPRESS,
    get_num_threads,
    set_num_threads,
)
from cvxpy.transforms import linearize, partial_optimize, suppfunc
from cvxpy.version import version as __version__  # cvxpy/version.py is auto-generated

__all__ = [
    "CallbackParam",
    "CBC",
    "CLARABEL",
    "Constant",
    "Constraint",
    "COPT",
    "CPLEX",
    "CPP_CANON_BACKEND",
    "CVXOPT",
    "DCPError",
    "DGPError",
    "DIFFCP",
    "disable_warnings",
    "DPPError",
    "ECOS_BB",
    "ECOS",
    "enable_warnings",
    "ExpCone",
    "Expression",
    "FiniteSet",
    "get_num_threads",
    "GLOP",
    "GLPK_MI",
    "GLPK",
    "GUROBI",
    "INFEASIBLE_INACCURATE",
    "INFEASIBLE",
    "installed_solvers",
    "linearize",
    "Maximize",
    "Minimize",
    "MOSEK",
    "NAG",
    "NonNeg",
    "NonPos",
    "Objective",
    "OpRelEntrConeQuad",
    "OPTIMAL_INACCURATE",
    "OPTIMAL",
    "OSQP",
    "Parameter",
    "partial_optimize",
    "PDLP",
    "PIQP",
    "PowCone3D",
    "PowConeND",
    "Problem",
    "PROXQP",
    "PSD",
    "RelEntrConeQuad",
    "ROBUST_KKTSOLVER",
    "RUST_CANON_BACKEND",
    "SCIP",
    "SCIPY_CANON_BACKEND",
    "SCIPY",
    "SCS",
    "SDPA",
    "set_num_threads",
    "SOC",
    "SOLVER_ERROR",
    "SolverError",
    "suppfunc",
    "UNBOUNDED_INACCURATE",
    "UNBOUNDED",
    "USER_LIMIT",
    "Variable",
    "warnings_enabled",
    "XPRESS",
    "Zero",
]
__all__.extend(cvxpy.atoms.__all__)
