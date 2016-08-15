"""
Copyright 2013 Steven Diamond, Eric Chu

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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.log import log
import scipy
import numpy as np


class log1p(log):
    """Elementwise :math:`\log (1 + x)`.
    """

    def __init__(self, x):
        super(log1p, self).__init__(x)

    @log.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise natural log of x+1.
        """
        return scipy.special.log1p(values[0])

    def sign_from_args(self):
        """The same sign as the argument.
        """
        return (self.args[0].is_positive(), self.args[0].is_negative())

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.size[0]*self.size[1]
        # Outside domain or on boundary.
        if np.min(values[0]) <= -1:
            # Non-differentiable.
            return [None]
        else:
            grad_vals = 1.0/(values[0]+1)
            return [log1p.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= -1]

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
        ones = lu.create_const(np.mat(np.ones(x.size)), x.size)
        xp1 = lu.sum_expr([x, ones])
        return log.graph_implementation([xp1], size, data)
