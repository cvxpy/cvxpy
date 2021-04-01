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

from cvxpy.atoms.elementwise.elementwise import Elementwise
from typing import Tuple

import scipy.special
import numpy as np

# TODO(akshayka): DGP support.


class huber(Elementwise):
    """The Huber function

    .. math::

        \\operatorname{Huber}(x, M) =
            \\begin{cases}
                2M|x|-M^2 & \\text{for } |x| \\geq |M| \\\\
                      |x|^2 & \\text{for } |x| \\leq |M|.
            \\end{cases}

    :math:`M` defaults to 1.

    Parameters
    ----------
    x : Expression
        The expression to which the huber function will be applied.
    M : Constant
        A scalar constant.
    """

    def __init__(self, x, M: int = 1) -> None:
        self.M = self.cast_to_const(M)
        super(huber, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values) -> float:
        """Returns the huber function applied elementwise to x.
        """
        return 2*scipy.special.huber(self.M.value, values[0])

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

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine.
        """
        return self.args[0].is_affine()

    def get_data(self):
        """Returns the parameter M.
        """
        return [self.M]

    def validate_arguments(self) -> None:
        """Checks that M >= 0 and is constant.
        """
        if not (self.M.is_nonneg() and
                self.M.is_scalar() and
                self.M.is_constant()):
            raise ValueError("M must be a non-negative scalar constant.")
        super(huber, self).validate_arguments()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size
        min_val = np.minimum(np.abs(values[0]), self.M.value)
        grad_vals = 2*np.multiply(np.sign(values[0]), min_val)
        return [huber.elemwise_grad_to_diag(grad_vals, rows, cols)]
