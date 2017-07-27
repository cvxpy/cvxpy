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
