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

# Constants for operators
PLUS = "+"
MINUS = "-"
MUL = "*"

# Constants for linear_ops
END_MUL = "end *"
END_MUL_OP = [(None, END_MUL, [])]

# Prefix for default named variables.
VAR_PREFIX = "var"

# Key for the constant term.
CONSTANT = "CONSTANT"

# Constraint types
EQ_CONSTR = "=="
INEQ_CONSTR = "<="

# Solver Constants
SOLVED = "solved"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
UNKNOWN = "solver_error"

# Map of solver status to cvxpy status.
CVXOPT = "cvxopt"
CVXOPT_STATUS = {'optimal': SOLVED,
                 'primal infeasible': INFEASIBLE, 
                 'dual infeasible': UNBOUNDED,
                 'unknown': UNKNOWN}
ECOS = "ecos"
ECOS_STATUS = {0: SOLVED,
               1: INFEASIBLE, 
               2: UNBOUNDED,
               3: UNKNOWN}
SOLVER_STATUS = {CVXOPT: CVXOPT_STATUS,
                 ECOS: ECOS_STATUS}

# Map of constraint types.
EQ,INEQ,SOC,SDP,NONLIN = range(5)