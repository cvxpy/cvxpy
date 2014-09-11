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
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.constraint import Constraint

class SDP(Constraint):
    """
    A semi-definite cone constraint:
        { symmetric A | x.T*A*x >= 0 for all x }
    (the set of all symmetric matrices such that the quadratic
    form x.T*A*x is positive for all x).

    Attributes:
        A: The matrix variable constrained to be semi-definite.
    """
    def __init__(self, A):
        self.A = A
        super(SDP, self).__init__()

    def __str__(self):
        return "SDP(%s)" % self.A

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats SDP constraints as inequalities for the solver.

        Parameters
        ----------
        eq_constr : list
            A list of the equality constraints in the canonical problem.
        leq_constr : list
            A list of the inequality constraints in the canonical problem.
        dims : dict
            A dict with the dimensions of the conic constraints.
        solver : str
            The solver being called.
        """
        # A == A.T
        eq_constr += self.__format[0]
        # 0 <= A
        leq_constr += self.__format[1]
        # Update dims.
        dims[s.EQ_DIM] += self.size[0]*self.size[1]
        dims[s.SDP_DIM].append(self.size[0])

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        eq_constr = lu.create_eq(self.A, lu.transpose(self.A))
        leq_constr = lu.create_geq(self.A)
        return ([eq_constr], [leq_constr])

    @property
    def size(self):
        """The dimensions of the semi-definite cone.
        """
        return self.A.size
