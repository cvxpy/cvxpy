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

class SOC(Constraint):
    """A second-order cone constraint, i.e., norm2(x) <= t.

    Attributes:
        t: The scalar part of the second-order constraint.
        x_elems: The elements of the vector part of the constraint.
    """
    def __init__(self, t, x_elems):
        self.t = t
        self.x_elems = x_elems
        super(SOC, self).__init__()

    def __str__(self):
        return "SOC(%s, %s)" % (self.t, self.x_elems)

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats SOC constraints as inequalities for the solver.

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
        leq_constr += self.__format[1]
        # Update dims.
        dims[s.SOC_DIM].append(self.size[0])

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        leq_constr = [lu.create_geq(self.t)]
        for elem in self.x_elems:
            leq_constr.append(lu.create_geq(elem))
        return ([], leq_constr)

    @property
    def size(self):
        """The dimensions of the second-order cone.
        """
        rows = 1
        for elem in self.x_elems:
            rows += elem.size[0]*elem.size[1]
        return (rows, 1)
