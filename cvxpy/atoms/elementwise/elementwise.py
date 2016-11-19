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

import abc
from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import scipy.sparse as sp


class Elementwise(Atom):
    """ Abstract base class for elementwise atoms. """
    __metaclass__ = abc.ABCMeta

    def size_from_args(self):
        """Size is the same as the sum of the arguments.
        """
        return u.shape.sum_shapes([arg.size for arg in self.args])

    def validate_arguments(self):
        """
        Verify that all the shapes are the same
        or can be promoted.
        """
        u.shape.sum_shapes([arg.size for arg in self.args])

    @staticmethod
    def elemwise_grad_to_diag(value, rows, cols):
        """Converts elementwise gradient into a diagonal matrix for Atom._grad()

        Args:
            value: A scalar or NumPy matrix.

        Returns:
            A SciPy CSC sparse matrix.
        """
        if not np.isscalar(value):
            value = value.A.ravel(order='F')
        return sp.dia_matrix((value, [0]), shape=(rows, cols)).tocsc()

    @staticmethod
    def _promote(arg, size):
        """Promotes the lin op if necessary.

        Parameters
        ----------
        arg : LinOp
            LinOp to promote.
        size : tuple
            The size desired.

        Returns
        -------
        tuple
            Promoted LinOp.
        """
        if arg.size != size:
            return lu.promote(arg, size)
        else:
            return arg
