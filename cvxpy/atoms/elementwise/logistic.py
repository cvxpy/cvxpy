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
from cvxpy.atoms.elementwise.exp import exp
import numpy as np


class logistic(Elementwise):
    """:math:`\log(1 + e^{x})`

    This is a special case of log(sum(exp)) that is evaluates to a vector rather
    than to a scalar which is useful for logistic regression.
    """

    def __init__(self, x):
        super(logistic, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, adds 1, and takes the log.
        """
        return np.logaddexp(0, values[0])

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

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
        exp_val = np.exp(values[0])
        grad_vals = exp_val/(1 + exp_val)
        return [logistic.elemwise_grad_to_diag(grad_vals, rows, cols)]

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
        t = lu.create_var(size)

        # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
        obj0, constr0 = exp.graph_implementation([lu.neg_expr(t)], size)
        obj1, constr1 = exp.graph_implementation([lu.sub_expr(x, t)], size)
        lhs = lu.sum_expr([obj0, obj1])
        ones = lu.create_const(np.mat(np.ones(size)), size)
        constr = constr0 + constr1 + [lu.create_leq(lhs, ones)]

        return (t, constr)
