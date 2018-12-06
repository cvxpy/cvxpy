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
USER_LIMIT = "user_limit"
SOLVER_ERROR = "solver_error"
# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE]
# Statuses that indicate the problem is infeasible or unbounded.
INF_OR_UNB = [INFEASIBLE, INFEASIBLE_INACCURATE,
              UNBOUNDED, UNBOUNDED_INACCURATE]
# Statuses that indicate an error.
ERROR = [USER_LIMIT, SOLVER_ERROR]

# Solver names.
CVXOPT = "CVXOPT"
GLPK = "GLPK"
GLPK_MI = "GLPK_MI"
CBC = "CBC"
CPLEX = "CPLEX"
ECOS = "ECOS"
ECOS_BB = "ECOS_BB"
SCS = "SCS"
SUPER_SCS = "SUPER_SCS"
GUROBI = "GUROBI"
OSQP = "OSQP"
CPLEX = "CPLEX"
ELEMENTAL = "ELEMENTAL"
MOSEK = "MOSEK"
JULIA_OPT = "JULIA_OPT"
XPRESS = "XPRESS"
SOLVERS = [ECOS, ECOS_BB, CVXOPT, GLPK,
           GLPK_MI, SCS, GUROBI, OSQP, CPLEX, ELEMENTAL,
           MOSEK, CBC, JULIA_OPT, XPRESS, SUPER_SCS]

# Xpress-specific items
XPRESS_IIS = "XPRESS_IIS"
XPRESS_TROW = "XPRESS_TROW"

# Parallel (meta) solver
PARALLEL = "parallel"

# Robust CVXOPT LDL KKT solver.
ROBUST_KKTSOLVER = "robust"

# Map of constraint types.
# TODO(akshayka): These should be defined in a solver module.
EQ, LEQ, SOC, SOC_EW, PSD, EXP, BOOL, INT = range(8)

# Keys in the dictionary of cone dimensions.
# TODO(akshayka): These should be defined in a solver module.
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
LOG_LOG_AFFINE = "LOG-LOG AFFINE"
LOG_LOG_CONVEX = "LOG-LOG CONVEX"
LOG_LOG_CONCAVE = "LOG-LOG CONCAVE"
ZERO = "ZERO"
NONNEG = "NONNEGATIVE"
NONPOS = "NONPOSITIVE"
UNKNOWN = "UNKNOWN"

# Numerical tolerances
EIGVAL_TOL = 1e-10
PSD_NSD_PROJECTION_TOL = 1e-8
GENERAL_PROJECTION_TOL = 1e-10
SPARSE_PROJECTION_TOL = 1e-10
