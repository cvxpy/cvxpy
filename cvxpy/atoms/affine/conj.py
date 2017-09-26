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

from cvxpy.atoms.affine.affine_atom import AffAtom
import numpy as np


class conj(AffAtom):
    """Complex conjugate.
    """
    def __init__(self, expr):
        super(conj, self).__init__(expr)

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        return np.conj(values[0])

    def shape_from_args(self):
        """Returns the shape of the expression.
        """
        return self.args[0].shape

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_symmetric(self):
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_hermitian(self):
        """Is the expression Hermitian?
        """
        return self.args[0].is_hermitian()
