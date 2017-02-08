"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.exponential import ExpCone
import numpy as np


class log(Elementwise):
    """Elementwise :math:`\log x`.
    """

    def __init__(self, x):
        super(log, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise natural log of x.
        """
        return np.log(values[0])

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

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
        if np.min(values[0]) <= 0:
            # Non-differentiable.
            return [None]
        else:
            grad_vals = 1.0/values[0]
            return [log.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0]

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
        x = arg_objs[0]
        ones = lu.create_const(np.mat(np.ones(size)), size)

        return (t, [ExpCone(t, ones, x)])
