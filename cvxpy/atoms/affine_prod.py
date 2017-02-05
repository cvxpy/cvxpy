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

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import numpy as np
import scipy.sparse as sp


class affine_prod(Atom):
    """Product of two affine expressions.
    """

    def __init__(self, x, y):
        super(affine_prod, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the product of two affine expressions.
        """
        return np.dot(values[0], values[1])

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return u.shape.mul_shapes(self.args[0].size, self.args[1].size)

    def sign_from_args(self):
        """Default to rules for times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_atom_convex(self):
        """Affine times affine is not convex
        """
        return False

    def is_atom_concave(self):
        """Affine times affine is not concave
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1-idx].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1-idx].is_negative()

    def validate_arguments(self):
        """Check dimensions of arguments and linearity.
        """
        if not self.args[0].is_affine() or not self.args[1].is_affine():
            raise ValueError("The arguments to affine_prod must be affine.")
        u.shape.mul_shapes(self.args[0].size, self.args[1].size)

    def is_quadratic(self):
        """Is the expression quadratic?
        """
        return True

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = values[0]
        Y = values[1]

        DX_rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.args[0].size[0]*self.args[1].size[1]

        # DX = [diag(Y11), diag(Y12), ...]
        #      [diag(Y21), diag(Y22), ...]
        #      [   ...        ...     ...]
        DX = sp.dok_matrix((DX_rows, cols))
        for k in range(self.args[0].size[0]):
            DX[k::self.args[0].size[0], k::self.args[0].size[0]] = Y
        DX = sp.csc_matrix(DX)
        DY = sp.block_diag([X.T for k in range(self.args[1].size[1])], 'csc')

        return [DX, DY]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        return NotImplemented
