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

import cvxpy.settings as s
from cvxpy.constraints.bool_constr import BoolConstr

class IntConstr(BoolConstr):
    """
    An integer constraint:
        X_{ij} in Z for all i,j.

    Attributes:
        noncvx_var: A variable constrained to be elementwise integral.
        lin_op: The linear operator equal to the noncvx_var.
    """
    CONSTR_TYPE = s.INT_IDS

    def __str__(self):
        return "IntConstr(%s)" % self.lin_op
