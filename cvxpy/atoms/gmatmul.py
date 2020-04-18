"""
Copyright, the CVXPY authors

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

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import numpy as np


class gmatmul(Atom):
    r"""Geometric matrix multiplication; :math:`A \mathbin{\diamond} X`.

    For :math:`A \in \mathbf{R}^{m \times n}` and
    :math:`X \in \mathbf{R}^{n \times p}_{++}`, this atom represents

    .. math::

        \left[\begin{array}{ccc}
         \prod_{j=1}^n X_{j1}^{A_{1j}} & \cdots & \prod_{j=1}^n X_{pj}^{A_{1j}} \\
         \vdots &  & \vdots \\
         \prod_{j=1}^n X_{j1}^{A_{mj}} & \cdots & \prod_{j=1}^n X_{pj}^{A_{mj}}
        \end{array}\right]

    This atom is log-log affine (in :math:`X`).

    Parameters
    ----------
    A : cvxpy.Expression
        A constant matrix.
    X : cvxpy.Expression
        A positive matrix.
    """
    def __init__(self, A, X):
        self.A = Atom.cast_to_const(A)
        super(gmatmul, self).__init__(X)

    def numeric(self, values):
        """Geometric matrix multiplication.
        """
        logX = np.log(values[0])
        return np.exp(self.A.value @ logX)

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.A,
                               self.args[0])

    def validate_arguments(self):
        """Raises an error if the arguments are invalid.
        """
        super(gmatmul, self).validate_arguments()
        if not self.A.is_constant():
            raise ValueError(
                "gmatmul(A, X) requires that A be constant."
            )
        if not self.args[0].is_pos():
            raise ValueError(
                "gmatmul(A, X) requires that X be positive."
            )

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.A.shape, self.args[0].shape)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.A]

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.A.is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.A.is_nonpos()

    def _grad(self, values):
        return None
