"""
Copyright 2013 Steven Diamond

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
from cvxpy.atoms.elementwise.elementwise import Elementwise
import numpy as np
if sys.version_info >= (3, 0):
    from functools import reduce


class maximum(Elementwise):
    """Elementwise maximum of a sequence of expressions.
    """

    def __init__(self, arg1, arg2, *args):
        """Requires at least 2 arguments.
        """
        super(maximum, self).__init__(arg1, arg2, *args)

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
        is_pos = any(arg.is_nonneg() for arg in self.args)
        is_neg = all(arg.is_nonpos() for arg in self.args)
        return (is_pos, is_neg)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self):
        return True

    def is_atom_log_log_concave(self):
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
        return all(arg.is_pwl() for arg in self.args)

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
            rows = self.args[idx].size
            cols = self.size
            grad_vals = (value == max_vals) & unused
            # Remove all the max_vals that were used.
            unused[value == max_vals] = 0
            grad_list += [maximum.elemwise_grad_to_diag(grad_vals,
                                                        rows, cols)]
        return grad_list
