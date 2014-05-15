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

from cvxpy.atoms.max_entries import max_entries
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu

class min_entries(max_entries):
    """:math:`\min_{i,j}\{X_{i,j}\}`.
    """
    def __init__(self, x):
        super(min_entries, self).__init__(x)

    @max_entries.numpy_numeric
    def numeric(self, values):
        """Returns the smallest entry in x.
        """
        return values[0].min()

    def func_curvature(self):
        """Default curvature is concave.
        """
        return u.Curvature.CONCAVE

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        x = arg_objs[0]
        t = lu.create_var((1, 1))
        promoted_t = lu.promote(t, x.size)
        constraints = [lu.create_leq(promoted_t, x)]
        return (t, constraints)
