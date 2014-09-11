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
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.utilities import format_elemwise

class SOC_Elemwise(SOC):
    """A second-order cone constraint for each element of the input.

    norm2([x1_ij; ... ; xn_ij]) <= t_ij for all i,j.

    Assumes t, xi, ..., xn all have the same dimensions.

    Attributes:
        t: The scalar part of the second-order constraint.
        x_elems: The elements of the vector part of the constraint.
    """
    def __str__(self):
        return "SOC_Elemwise(%s, %s)" % (self.t, self.x_elems)

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
        for cone_size in self.size:
            dims[s.SOC_DIM].append(cone_size[0])

    @pu.lazyprop
    def __format(self):
        """Internal version of format with cached results.

        Returns
        -------
        tuple
            (equality constraints, inequality constraints)
        """
        return ([], format_elemwise([self.t] + self.x_elems))

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.t.size[0]*self.t.size[1]

    def cone_size(self):
        """The dimensions of a single cone.
        """
        return (1 + len(self.x_elems), 1)

    @property
    def size(self):
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the dimensions of the elementwise cones.
        """
        cones = []
        cone_size = self.cone_size()
        for i in range(self.num_cones()):
            cones.append(cone_size)
        return cones
