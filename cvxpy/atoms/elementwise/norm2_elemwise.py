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
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints import SOC_Elemwise
import numpy as np
from numpy import linalg as LA

class norm2_elemwise(Elementwise):
    """Groups corresponding elements and takes the L2 norm.
    """

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Stack the values and take the L2 norms of the columns.
        """
        rows, cols = self.size
        mat_3D = np.zeros((rows, cols, len(values)))
        for i in range(len(values)):
            mat_3D[:, :, i] = values[i]
        return LA.norm(mat_3D, axis=2)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        return len(self.args)*[u.monotonicity.SIGNED]

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
        t = lu.create_var(size)
        for i, obj in enumerate(arg_objs):
            # Promote obj.
            if obj.size != size:
                arg_objs[i] = lu.promote(obj, size)

        return (t, [SOC_Elemwise(t, arg_objs)])
