"""
Copyright 2013 Steven Diamond

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
import logging
import sys

LOGGER = logging.getLogger("__cvxpy__")
LOGGER.propagate = False
LOGGER.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="(CVXPY) %(asctime)s: %(message)s", datefmt="%b %d %I:%M:%S %p"
)
_stream_handler.setFormatter(_formatter)
LOGGER.addHandler(_stream_handler)


# Constants for operators.
PLUS = "+"
MINUS = "-"
MUL = "*"

# Prefix for default named variables.
VAR_PREFIX = "var"
# Prefix for default named parameters.
PARAM_PREFIX = "param"

# Used to trick Numpy so cvxpy can overload ==.
NP_EQUAL_STR = "equal"

# Constraint types
EQ_CONSTR = "=="
INEQ_CONSTR = "<="

# Solver Constants
OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal_inaccurate"
INFEASIBLE = "infeasible"
INFEASIBLE_INACCURATE = "infeasible_inaccurate"
UNBOUNDED = "unbounded"
UNBOUNDED_INACCURATE = "unbounded_inaccurate"
INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
USER_LIMIT = "user_limit"
SOLVER_ERROR = "solver_error"
# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE, USER_LIMIT]
# Statuses that indicate the problem is infeasible or unbounded.
INF_OR_UNB = [INFEASIBLE, INFEASIBLE_INACCURATE,
              UNBOUNDED, UNBOUNDED_INACCURATE,
              INFEASIBLE_OR_UNBOUNDED]
# Statuses that indicate an inaccurate solution.
INACCURATE = [OPTIMAL_INACCURATE, INFEASIBLE_INACCURATE,
              UNBOUNDED_INACCURATE, USER_LIMIT]
# Statuses that indicate an error.
ERROR = [SOLVER_ERROR]

# Solver names.
CVXOPT = "CVXOPT"
GLPK = "GLPK"
GLPK_MI = "GLPK_MI"
GLOP = "GLOP"
CBC = "CBC"
COPT = "COPT"
ECOS = "ECOS"
ECOS_BB = "ECOS_BB"
SCS = "SCS"
SDPA = "SDPA"
DIFFCP = "DIFFCP"
GUROBI = "GUROBI"
OSQP = "OSQP"
PIQP = "PIQP"
PROXQP = "PROXQP"
QOCO = "QOCO"
CPLEX = "CPLEX"
MOSEK = "MOSEK"
XPRESS = "XPRESS"
NAG = "NAG"
PDLP = "PDLP"
SCIP = "SCIP"
SCIPY = "SCIPY"
CLARABEL = "CLARABEL"
DAQP = "DAQP"
HIGHS = "HIGHS"
SOLVERS = [CLARABEL, ECOS, CVXOPT, GLOP, GLPK, GLPK_MI,
           SCS, SDPA, GUROBI, OSQP, CPLEX,
           MOSEK, CBC, COPT, XPRESS, PIQP, PROXQP, QOCO,
           NAG, PDLP, SCIP, SCIPY, DAQP, HIGHS]

# Xpress-specific items
XPRESS_IIS = "XPRESS_IIS"
XPRESS_TROW = "XPRESS_TROW"

# Parameterized problem.
PARAM_PROB = "param_prob"

# Parallel (meta) solver.
PARALLEL = "parallel"

# Robust CVXOPT LDL KKT solver.
ROBUST_KKTSOLVER = "robust"

# Map of constraint types.
# TODO(akshayka): These should be defined in a solver module.
EQ, LEQ, SOC, SOC_EW, PSD, EXP, BOOL, INT = range(8)

# Keys in the dictionary of cone dimensions.
# TODO(akshayka): These should be defined in a solver module.
#   Riley follow-up on this: cone dims are now defined in matrix
#   stuffing modules (e.g. cone_matrix_stuffing.py), rather than
#   the solver module.

EQ_DIM = "f"
LEQ_DIM = "l"
SOC_DIM = "q"
PSD_DIM = "s"
EXP_DIM = "ep"
# Keys for non-convex constraints.
BOOL_IDS = "bool_ids"
BOOL_IDX = "bool_idx"
INT_IDS = "int_ids"
INT_IDX = "int_idx"
LOWER_BOUNDS = "lower_bounds"
UPPER_BOUNDS = "upper_bounds"

# Keys for results_dict.
STATUS = "status"
VALUE = "value"
OBJ_OFFSET = "obj_offset"
PRIMAL = "primal"
EQ_DUAL = "eq_dual"
INEQ_DUAL = "ineq_dual"
SOLVE_TIME = "solve_time"  # in seconds
SETUP_TIME = "setup_time"  # in seconds
NUM_ITERS = "num_iters"  # number of iterations
EXTRA_STATS = "solver_specific_stats"

# Keys for problem data dict.
C = "c"
OFFSET = "offset"
P = "P"
Q = "q"
A = "A"
B = "b"
G = "G"
H = "h"
F = "F"
DIMS = "dims"
BOOL_IDX = "bool_vars_idx"
INT_IDX = "int_vars_idx"

# Keys for curvature and sign.
CONSTANT = "CONSTANT"
AFFINE = "AFFINE"
CONVEX = "CONVEX"
CONCAVE = "CONCAVE"
QUASILINEAR = "QUASILINEAR"
QUASICONVEX = "QUASICONVEX"
QUASICONCAVE = "QUASICONCAVE"
LOG_LOG_CONSTANT = "LOG-LOG CONSTANT"
LOG_LOG_AFFINE = "LOG-LOG AFFINE"
LOG_LOG_CONVEX = "LOG-LOG CONVEX"
LOG_LOG_CONCAVE = "LOG-LOG CONCAVE"
ZERO = "ZERO"
NONNEG = "NONNEGATIVE"
NONPOS = "NONPOSITIVE"
UNKNOWN = "UNKNOWN"

# Canonicalization backends
NUMPY_CANON_BACKEND = "NUMPY"
SCIPY_CANON_BACKEND = "SCIPY"
RUST_CANON_BACKEND = "RUST"
CPP_CANON_BACKEND = "CPP"

# Default canonicalization backend, pyodide uses SciPy
DEFAULT_CANON_BACKEND = CPP_CANON_BACKEND if sys.platform != "emscripten" else SCIPY_CANON_BACKEND

# Numerical tolerances
EIGVAL_TOL = 1e-10
PSD_NSD_PROJECTION_TOL = 1e-8
GENERAL_PROJECTION_TOL = 1e-10
SPARSE_PROJECTION_TOL = 1e-10
ATOM_EVAL_TOL = 1e-4
CHOL_SYM_TOL = 1e-14

# DPP is slow when total size of parameters
# exceed this threshold.
PARAM_THRESHOLD = 1e4

# threads to use during compilation
# -1 defaults to system default (configurable via the OMP_NUM_THREADS
# environment variable)
NUM_THREADS = -1

# Flag to allow ND expressions.
ALLOW_ND_EXPR = True

PRINT_EDGEITEMS = 2
PRINT_THRESHOLD = 5


def set_num_threads(num_threads: int) -> None:
    global NUM_THREADS
    NUM_THREADS = num_threads


def get_num_threads() -> int:
    return NUM_THREADS
