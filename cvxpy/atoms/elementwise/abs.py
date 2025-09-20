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

from typing import Tuple

import numpy as np

from .elementwise import Elementwise


class abs(Elementwise):
    """ Elementwise absolute value """
    _allow_complex = True

    def __init__(self, x) -> None:
        super(abs, self).__init__(x)

    # Returns the elementwise absolute value of x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.absolute(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[idx].is_nonpos()

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return self.args[0].is_pwl() and \
            (self.args[0].is_real() or self.args[0].is_imag())

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad: +1 if positive, -1 if negative.
        rows = self.expr.size
        cols = self.size
        D = np.zeros(self.expr.shape)
        D += (values[0] > 0)
        D -= (values[0] < 0)
        return [abs.elemwise_grad_to_diag(D, rows, cols)]
