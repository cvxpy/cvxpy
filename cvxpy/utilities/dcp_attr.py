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

import bool_mat_utils as bu
from curvature import Curvature
import key_utils as ku
from shape import Shape
from sign import Sign

class DCPAttr(object):
    """ A data structure for the sign, curvature, and shape of an expression.

    Attributes:
        sign: The signs of the entries in the matrix expression.
        curvature: The curvatures of the entries in the matrix expression.
        shape: The dimensions of the matrix expression.
    """

    def __init__(self, sign, curvature, shape):
        self.sign = sign
        self.curvature = curvature
        self.shape = shape

    def __getitem__(self, key):
        """Determines the DCP attributes of an index/slice.

        Args:
            key: A (slice, slice) tuple.

        Returns:
            The DCPAttr of the index/slice into the matrix expression.
        """
        shape = Shape(*ku.size(key, self.shape))

        # Reduce 1x1 matrices to scalars.
        neg_mat = bu.to_scalar(bu.index(self.sign.neg_mat, key))
        pos_mat = bu.to_scalar(bu.index(self.sign.pos_mat, key))
        cvx_mat = bu.to_scalar(bu.index(self.curvature.cvx_mat, key))
        conc_mat = bu.to_scalar(bu.index(self.curvature.conc_mat, key))
        nonconst_mat = bu.to_scalar(bu.index(self.curvature.nonconst_mat, key))

        return DCPAttr(Sign(neg_mat, pos_mat),
                       Curvature(cvx_mat, conc_mat, nonconst_mat),
                       shape)

    @property
    def T(self):
        """Determines the DCP attributes of a transpose.

        Returns:
            The DCPAttr of the transpose of the matrix expression.
        """
        rows, cols = self.shape.size
        shape = Shape(cols, rows)

        neg_mat = self.sign.neg_mat.T
        pos_mat = self.sign.pos_mat.T
        cvx_mat = self.curvature.cvx_mat.T
        conc_mat = self.curvature.conc_mat.T
        nonconst_mat = self.curvature.nonconst_mat.T

        return DCPAttr(Sign(neg_mat, pos_mat),
                       Curvature(cvx_mat, conc_mat, nonconst_mat),
                       shape)

    def __add__(self, other):
        """Determines the DCP attributes of two expressions added together.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the sum.
        """
        shape = self.shape + other.shape
        sign = self.sign + other.sign
        curvature = self.curvature + other.curvature
        return DCPAttr(sign, curvature, shape)

    def __sub__(self, other):
        """Determines the DCP attributes of one expression minus another.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the difference.
        """
        shape = self.shape + other.shape
        sign = self.sign - other.sign
        curvature = self.curvature - other.curvature
        return DCPAttr(sign, curvature, shape)

    def __mul__(self, other):
        """Determines the DCP attributes of two expressions multiplied together.

        Assumes one of the arguments has constant curvature.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        shape = self.shape * other.shape
        lh_sign = self.sign.promote(*self.shape.size)
        rh_sign = other.sign.promote(*other.shape.size)
        sign = lh_sign * rh_sign
        rh_curvature = other.curvature.promote(*other.shape.size)
        curvature = Curvature.sign_mul(lh_sign, rh_curvature)
        return DCPAttr(sign, curvature, shape)

    def __div__(self, other):
        """Determines the DCP attributes of one expression divided by another.

        Assumes one of the arguments has constant curvature.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        return other*self

    def __neg__(self):
        """Determines the DCP attributes of a negated expression.
        """
        return DCPAttr(-self.sign, -self.curvature, self.shape)
