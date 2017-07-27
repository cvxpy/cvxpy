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
