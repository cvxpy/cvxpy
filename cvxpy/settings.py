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
SOLVER_ERROR = "solver_error"
# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE]
# Statuses that indicate the problem is infeasible or unbounded.
INF_OR_UNB = [INFEASIBLE, INFEASIBLE_INACCURATE,
              UNBOUNDED, UNBOUNDED_INACCURATE]

# Map of solver status to cvxpy status.
CVXOPT = "CVXOPT"
CVXOPT_STATUS = {'optimal': OPTIMAL,
                 'primal infeasible': INFEASIBLE,
                 'dual infeasible': UNBOUNDED,
                 'unknown': SOLVER_ERROR}
ECOS = "ECOS"
ECOS_STATUS = {0: OPTIMAL,
               1: INFEASIBLE,
               2: UNBOUNDED,
               10: OPTIMAL_INACCURATE,
               11: INFEASIBLE_INACCURATE,
               12: UNBOUNDED_INACCURATE,
               -1: SOLVER_ERROR,
               -2: SOLVER_ERROR,
               -3: SOLVER_ERROR,
               -7: SOLVER_ERROR}

SCS = "SCS"
SCS_STATUS = {"Solved": OPTIMAL,
              "Solved/Inaccurate": OPTIMAL_INACCURATE,
              "Unbounded": UNBOUNDED,
              "Unbounded/Inaccurate": UNBOUNDED_INACCURATE,
              "Infeasible": INFEASIBLE,
              "Infeasible/Inaccurate": INFEASIBLE_INACCURATE,
              "Failure": SOLVER_ERROR,
              "Indeterminate": SOLVER_ERROR}

SOLVER_STATUS = {CVXOPT: CVXOPT_STATUS,
                 ECOS: ECOS_STATUS,
                 SCS: SCS_STATUS}

# Solver capabilities.
SDP_CAPABLE = [CVXOPT, SCS]
EXP_CAPABLE = [CVXOPT, SCS]
SOCP_CAPABLE = [ECOS, CVXOPT, SCS]

# Map of constraint types.
EQ, LEQ, SOC, SOC_EW, SDP, EXP = range(6)

# Keys in the dictionary of cone dimensions.
EQ_DIM = "f"
LEQ_DIM = "l"
SOC_DIM = "q"
SDP_DIM = "s"
EXP_DIM = "ep"
