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
ECOS = "ECOS"
ECOS_BB = "ECOS_BB"
SCS = "SCS"
GUROBI = "GUROBI"
OSQP = "OSQP"
CPLEX = "CPLEX"
ELEMENTAL = "ELEMENTAL"
MOSEK = "MOSEK"
JULIA_OPT = "JULIA_OPT"
XPRESS = "XPRESS"
SOLVERS = [ECOS, ECOS_BB, CVXOPT, GLPK,
           GLPK_MI, SCS, GUROBI, OSQP, CPLEX, ELEMENTAL,
           MOSEK, CBC, JULIA_OPT, XPRESS]

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
ZERO = "ZERO"
NONNEG = "NONNEGATIVE"
NONPOS = "NONPOSITIVE"
UNKNOWN = "UNKNOWN"
