"""
Copyright 2013 Steven Diamond

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

from scipy import linalg as LA
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.sum_largest import sum_largest


class lambda_sum_largest(lambda_max):
    """Sum of the largest k eigenvalues.
    """
    _allow_complex = True

    def __init__(self, X, k):
        self.k = k
        super(lambda_sum_largest, self).__init__(X)

    def validate_arguments(self):
        """Verify that the argument A is square.
        """
        X = self.args[0]
        if not X.ndim == 2 or X.shape[0] != X.shape[1]:
            raise ValueError("First argument must be a square matrix.")
        elif int(self.k) != self.k or self.k <= 0:
            raise ValueError("Second argument must be a positive integer.")

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        eigs = LA.eigvals(values[0])
        return sum_largest(eigs, self.k).value

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return NotImplemented
