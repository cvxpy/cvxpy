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

import sys
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
import numpy as np
if sys.version_info >= (3, 0):
    from functools import reduce


class max_elemwise(Elementwise):
    """ Elementwise maximum. """

    def __init__(self, *args):
        """Requires at least 2 arguments.
        """
        if len(args) == 0 or (len(args) == 1 and not isinstance(args[0], list)):
            raise TypeError("max_elemwise requires at least two arguments or a list.")
        elif len(args) == 1:
            args = args[0]
        super(max_elemwise, self).__init__(*args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise maximum.
        """
        return reduce(np.maximum, values)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Reduces the list of argument signs according to the following rules:
        #     POSITIVE, ANYTHING = POSITIVE
        #     ZERO, UNKNOWN = POSITIVE
        #     ZERO, ZERO = ZERO
        #     ZERO, NEGATIVE = ZERO
        #     UNKNOWN, NEGATIVE = UNKNOWN
        #     NEGATIVE, NEGATIVE = NEGATIVE
        is_pos = any([arg.is_positive() for arg in self.args])
        is_neg = all([arg.is_negative() for arg in self.args])
        return (is_pos, is_neg)

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

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return all([arg.is_pwl() for arg in self.args])

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        max_vals = np.matrix(self.numeric(values))
        unused = np.matrix(np.ones(max_vals.shape), dtype=bool)
        grad_list = []
        for idx, value in enumerate(values):
            rows = self.args[idx].size[0]*self.args[idx].size[1]
            cols = self.size[0]*self.size[1]
            grad_vals = (value == max_vals) & unused
            # Remove all the max_vals that were used.
            unused[value == max_vals] = 0
            grad_list += [max_elemwise.elemwise_grad_to_diag(grad_vals,
                                                             rows, cols)]
        return grad_list

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
        constraints = []
        for obj in arg_objs:
            obj = Elementwise._promote(obj, size)
            constraints.append(lu.create_leq(obj, t))
        return (t, constraints)
