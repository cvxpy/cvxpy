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

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.affine.reshape import reshape
import numpy as np
import scipy.sparse as sp
import scipy as scipy


def quad_over_lin(x, y, axis=0):

    if y.is_scalar():
        x = reshape(x, (x.size, 1))
        y = reshape(y, (1,))
    elif y.is_vector():
        if x.is_vector():
            if x.size != y.size:
                raise ValueError(
                    "If both arguments to quad_over_lin are vectors, their sizes must match."
                )
            else:
                x = reshape(x, (1, x.size))
                y = reshape(y, (y.size,))
        elif x.is_matrix():
            x = x.T if axis == 1 else x
            if x.shape[1] != y.size:
                raise ValueError(
                    "For quad_over_lin(X, y, axis) with matrix X and vector y, "
                    "we must have X.shape[1-axis] == y.size"
                )
        else:
            raise ValueError(
                "If the second argument to quad_over_lin is a vector,"
                "the first argument must be a vector or matrix."
            )

    return QuadOverLin(x, y)


class QuadOverLin(Atom):
    """ :math:`(sum_{ij}X^2_{ij})/y`

    """
    _allow_complex = True

    def __init__(self, x, y):
        super(QuadOverLin, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        if self.args[0].is_complex():
            return (np.square(values[0].imag) + np.square(values[0].real)) @ (1/values[1])
        return np.square(values[0]) @ (1/values[1])
    
    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        # y > 0.
        return [self.args[1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = values[0]
        y = values[1]
        if y <= 0:
            return [None, None]
        else:
            # DX = 2X/y, Dy = -||X||^2_2/y^2
            if self.args[0].is_complex():
                Dy = -(np.square(X.real) + np.square(X.imag)).sum()/np.square(y)
            else:
                Dy = -np.square(X).sum()/np.square(y)

            Dy = sp.csc_matrix(Dy)
            DX = 2.0*X/y
            DX = np.reshape(DX, (self.args[0].size, 1))
            DX = scipy.sparse.csc_matrix(DX)
            return [DX, Dy]

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        if self.args[0].shape[1] != self.args[1].size:
            raise ValueError(
                "For quad_over_lin(X, y, axis) with matrix X and vector y, "
                "we must have X.shape[1-axis] == y.size"
            )

        if self.args[1].is_complex():
            raise ValueError("The second argument to QuadOverLin cannot be complex.")
        super(QuadOverLin, self).validate_arguments()

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return self.args[1].shape

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

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
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return (idx == 0) and self.args[idx].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return ((idx == 0) and self.args[idx].is_nonpos()) or (idx == 1)

    def is_quadratic(self):
        """Quadratic if x is affine and y is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def is_qpwa(self):
        """Quadratic of piecewise affine if x is PWL and y is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()
