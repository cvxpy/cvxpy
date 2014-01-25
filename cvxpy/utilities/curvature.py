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
import numpy as np

class Curvature(object):
    """Curvatures of the entries in an expression.

    Attributes:
        cvx_mat: A boolean matrix indicating whether each entry is convex.
        conc_mat: A boolean matrix indicating whether each entry is concave.
        nonconst_mat: A boolean matrix indicating whether each entry is
                      nonconstant.
    """

    CONVEX_KEY = 'CONVEX'
    CONCAVE_KEY = 'CONCAVE'
    UNKNOWN_KEY = 'UNKNOWN'
    AFFINE_KEY = 'AFFINE'
    CONSTANT_KEY = 'CONSTANT'

    # Map of curvature string to scalar (cvx_mat, conc_mat) values.
    # Affine is (False, False) and unknown is (True, True).
    CURVATURE_MAP = {
        CONVEX_KEY: (np.bool_(True),
                     np.bool_(False),
                     np.bool_(True)),
        CONCAVE_KEY: (np.bool_(False),
                      np.bool_(True),
                      np.bool_(True)),
        UNKNOWN_KEY: (np.bool_(True),
                      np.bool_(True),
                      np.bool_(True)),
        AFFINE_KEY: (np.bool_(False),
                     np.bool_(False),
                     np.bool_(True)),
        CONSTANT_KEY: (np.bool_(False),
                       np.bool_(False),
                       np.bool_(False)),
    }

    def __init__(self, cvx_mat, conc_mat, nonconst_mat):
        self.cvx_mat = cvx_mat
        self.conc_mat = conc_mat
        self.nonconst_mat = nonconst_mat

    @staticmethod
    def name_to_curvature(curvature_str):
        """Converts a curvature name to a Curvature object.

        Args:
            curvature_str: A key in the CURVATURE_MAP.

        Returns:
            A Curvature initialized with the selected value from CURVATURE_MAP.
        """
        curvature_str = curvature_str.upper()
        if curvature_str in Curvature.CURVATURE_MAP:
            return Curvature(*Curvature.CURVATURE_MAP[curvature_str])
        else:
            raise Exception("'%s' is not a valid curvature name." %
                            str(curvature_str))

    def is_constant(self):
        """Is the expression constant?
        """
        return not self.nonconst_mat.any()

    def is_affine(self):
        """Is the expression affine?
        """
        return not (self.cvx_mat | self.conc_mat).any()

    def is_convex(self):
        """Is the expression convex?
        """
        return not self.conc_mat.any()

    def is_concave(self):
        """Is the expression concave?
        """
        return not self.cvx_mat.any()

    def is_dcp(self):
        """Is the expression DCP compliant? (i.e., no unknown curvatures).
        """
        return not (self.cvx_mat & self.conc_mat).any()

    def __add__(self, other):
        """Handles the logic of adding curvatures.

        Cases:
          CONSTANT + ANYTHING = ANYTHING
          AFFINE + NONCONSTANT = NONCONSTANT
          CONVEX + CONCAVE = UNKNOWN
          SAME + SAME = SAME

        Args:
            self: The Curvature of the left-hand summand.
            other: The Curvature of the right-hand summand.

        Returns:
            The Curvature of the sum.
        """
        return Curvature(
            self.cvx_mat | other.cvx_mat,
            self.conc_mat | other.conc_mat,
            self.nonconst_mat | other.nonconst_mat
        )

    def __sub__(self, other):
        return self + -other

    @staticmethod
    def sign_mul(sign, curv):
        """Handles logic of sign by curvature multiplication.

        Cases:
            ZERO * ANYTHING = CONSTANT
            NON-ZERO * AFFINE/CONSTANT = AFFINE/CONSTANT
            UNKNOWN * NON-AFFINE = UNKNOWN
            POSITIVE * ANYTHING = ANYTHING
            NEGATIVE * CONVEX = CONCAVE
            NEGATIVE * CONCAVE = CONVEX

        Args:
            sign: The Sign of the left-hand multiplier.
            curv: The Curvature of the right-hand multiplier.

        Returns:
            The Curvature of the product.
        """
        cvx_mat = bu.dot(sign.pos_mat, curv.cvx_mat) | \
                  bu.dot(sign.neg_mat, curv.conc_mat)
        conc_mat = bu.dot(sign.pos_mat, curv.conc_mat) | \
                   bu.dot(sign.neg_mat, curv.cvx_mat)
        nonconst_mat = bu.dot(sign.pos_mat, curv.nonconst_mat) | \
                       bu.dot(sign.neg_mat, curv.nonconst_mat)
        # Simplify 1x1 matrices to scalars.
        cvx_mat = bu.to_scalar(cvx_mat)
        conc_mat = bu.to_scalar(conc_mat)
        nonconst_mat = bu.to_scalar(nonconst_mat)
        return Curvature(cvx_mat, conc_mat, nonconst_mat)

    def __neg__(self):
        """Equivalent to NEGATIVE * self.
        """
        return Curvature(self.conc_mat, self.cvx_mat, self.nonconst_mat)

    def __eq__(self, other):
        """Checks equality of arguments' attributes.
        """
        return np.all(self.cvx_mat == other.cvx_mat) and \
               np.all(self.conc_mat == other.conc_mat) and \
               np.all(self.nonconst_mat == other.nonconst_mat)

    def promote(self, rows, cols, keep_scalars=True):
        """Promotes the Curvature's internal matrices to the desired size.

        Args:
            rows: The number of rows in the promoted internal matrices.
            cols: The number of columns in the promoted internal matrices.
            keep_scalars: Don't convert scalars to matrices.
        """
        cvx_mat = bu.promote(self.cvx_mat, rows, cols, keep_scalars)
        conc_mat = bu.promote(self.conc_mat, rows, cols, keep_scalars)
        nonconst_mat = bu.promote(self.nonconst_mat, rows, cols, keep_scalars)
        return Curvature(cvx_mat, conc_mat, nonconst_mat)

    def __repr__(self):
        return "Curvature(%s, %s, %s)" % (repr(self.cvx_mat),
                                          repr(self.conc_mat),
                                          repr(self.nonconst_mat))

    def __str__(self):
        return "Curvature(%s, %s, %s)" % (self.cvx_mat,
                                          self.conc_mat,
                                          self.nonconst_mat)

    def get_readable_repr(self, rows, cols):
        """Converts the internal representation to a matrix of strings.

        Args:
            rows: The number of rows in the expression.
            cols: The number of columns in the expression.

        Returns:
            A curvature string or a Numpy 2D array of curvature strings.
        """
        curvature = self.promote(rows, cols, False)
        readable_mat = np.empty((rows, cols), dtype="object")
        for i in xrange(rows):
            for j in xrange(cols):
                # Is the entry constant?
                if not curvature.nonconst_mat[i, j]:
                    readable_mat[i, j] = self.CONSTANT_KEY
                # Is the entry unknown?
                elif curvature.cvx_mat[i, j] and \
                     curvature.conc_mat[i, j]:
                    readable_mat[i, j] = self.UNKNOWN_KEY
                # Is the entry convex?
                elif curvature.cvx_mat[i, j]:
                    readable_mat[i, j] = self.CONVEX_KEY
                # Is the entry concave?
                elif curvature.conc_mat[i, j]:
                    readable_mat[i, j] = self.CONCAVE_KEY
                # The entry is affine.
                else:
                    readable_mat[i, j] = self.AFFINE_KEY

        # Reduce readable_mat to a single string if homogeneous.
        if (readable_mat == readable_mat[0, 0]).all():
            return readable_mat[0, 0]
        else:
            return readable_mat

# Scalar curvatures.
Curvature.CONVEX = Curvature.name_to_curvature(Curvature.CONVEX_KEY)
Curvature.CONCAVE = Curvature.name_to_curvature(Curvature.CONCAVE_KEY)
Curvature.UNKNOWN = Curvature.name_to_curvature(Curvature.UNKNOWN_KEY)
Curvature.AFFINE = Curvature.name_to_curvature(Curvature.AFFINE_KEY)
Curvature.CONSTANT = Curvature.name_to_curvature(Curvature.CONSTANT_KEY)
