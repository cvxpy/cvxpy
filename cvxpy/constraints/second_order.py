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

import cvxpy.utilities as u
import cvxpy.settings as s
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.utilities import format_axis


class SOC(u.Canonical, Constraint):
    """A second-order cone constraint for each row/column.

    Assumes t is a vector the same length as X's columns (rows) for axis==0 (1).

    Attributes:
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.
    """

    def __init__(self, t, X, axis=0):
        assert t.shape[1] == 1
        self.axis = axis
        super(SOC, self).__init__([t, X])

    def __str__(self):
        return "SOC(%s, %s)" % (self.args[0], self.args[1])

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
        dims[s.SOC_DIM] += self.cone_sizes()

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        return ([], format_axis(self.args[0], self.args[1], self.axis))

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.args[0].shape[0]*self.args[0].shape[1]

    @property
    def size(self):
        """The number of entries in the combined cones.
        """
        # TODO use size of dual variable(s) instead.
        return sum(self.cone_sizes())

    def cone_sizes(self):
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        cones = []
        cone_size = 1 + self.args[1].shape[self.axis]
        for i in range(self.num_cones()):
            cones.append(cone_size)
        return cones

    def is_dcp(self):
        """Is the constraint DCP?
        """
        return all([arg.is_affine() for arg in self.args])

    #TODO hack
    def canonicalize(self):
        constr = []
        t, t_cons = self.args[0].canonical_form
        X, X_cons = self.args[1].canonical_form
        new_soc = SOC(t, X, self.axis)
        return (None, [new_soc] + t_cons + X_cons)
