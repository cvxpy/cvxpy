"""
Copyright 2022, the CVXPY authors

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
from typing import List, Tuple

from scipy import linalg as LA

from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint


class condition_number(Atom):
    """ Condition Number; :math:`\\lambda_{\\max}(A) / \\lambda_{\\min}(A)`.
        Requires that A be a Positive Semidefinite Matrix.
    """
    def __init__(self, A) -> None:
        super(condition_number, self).__init__(A)

    def numeric(self, values):
        """Returns the condition number of A.

        Requires that A be a Positive Semidefinite Matrix.
        """
        lo = hi = self.args[0].shape[0]-1
        max_eigen = LA.eigvalsh(values[0], eigvals=(lo, hi))[0]
        min_eigen = -LA.eigvalsh(-values[0], eigvals=(lo, hi))[0]
        return max_eigen / min_eigen

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0].H == self.args[0], self.args[0] >> 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        raise NotImplementedError

    def validate_arguments(self) -> None:
        """Verify that the argument A is square.
        """
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError(
                f"The argument {self.args[0].name()} to condition_number must be a square matrix."
            )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False
