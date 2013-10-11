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

from bool_mat import BoolMat
import bool_mat_utils as bu

class Curvature(object):
    """ 
    Curvatures of the entries in an expression.
    """
    CONVEX_KEY = 'CONVEX'
    CONCAVE_KEY = 'CONCAVE'
    UNKNOWN_KEY = 'UNKNOWN'
    AFFINE_KEY = 'AFFINE'
    CONSTANT_KEY = 'CONSTANT'

    # Map of curvature string to scalar (cvx_mat, conc_mat) values.
    # Affine is (False, False) and unknown is (True, True).
    CURVATURE_MAP = {
        CONVEX_KEY: (True, False, False),
        CONCAVE_KEY: (False, True, False),
        UNKNOWN_KEY: (True, True, False),
        AFFINE_KEY: (False, False, False),
        CONSTANT_KEY: (False, False, True),
    }

    # cvx_mat - a boolean matrix indicating whether each entry is convex.
    # conc_mat - a boolean matrix indicating whether each entry is concave.
    # constant - a boolean indicating whether the overall expression is constant.
    def __init__(self, cvx_mat, conc_mat, constant):
        self.cvx_mat = cvx_mat
        self.conc_mat = conc_mat
        self.constant = constant

    @staticmethod
    def name_to_curvature(curvature_str):
        curvature_str = curvature_str.upper()
        if curvature_str in Curvature.CURVATURE_MAP:
            return Curvature(*Curvature.CURVATURE_MAP[curvature_str])
        else:
            raise Exception("'%s' is not a valid curvature name." % str(curvature_str))

    # Is the expression constant?
    def is_constant(self):
        return self.constant

    # Is the expression affine?
    def is_affine(self):
        return not bu.any(self.cvx_mat | self.conc_mat)

    # Is the expression convex?
    def is_convex(self):
        return not bu.any(self.conc_mat)

    # Is the expression concave?
    def is_concave(self):
        return not bu.any(self.cvx_mat)

    # Is the expression DCP compliant? (i.e. no unknown curvatures)
    def is_dcp(self):
        return not bu.any(self.cvx_mat & self.conc_mat)

    # Arithmetic operators.
    """
    Resolves the logic of adding curvatures.
      CONSTANT + ANYTHING = ANYTHING
      AFFINE + NONCONSTANT = NONCONSTANT
      CONVEX + CONCAVE = UNKNOWN
      SAME + SAME = SAME
    """
    def __add__(self, other):
        return Curvature(
            self.cvx_mat | other.cvx_mat,
            self.conc_mat | other.conc_mat,
            self.constant & other.constant
        )
    
    def __sub__(self, other):
        return self + -other
    
    """
    Handles logic of sign by curvature multiplication:
        ZERO * ANYTHING = AFFINE/CONSTANT
        NON-ZERO * AFFINE/CONSTANT = AFFINE/CONSTANT
        UNKNOWN * NON-AFFINE = UNKNOWN
        POSITIVE * ANYTHING = ANYTHING
        NEGATIVE * CONVEX = CONCAVE
        NEGATIVE * CONCAVE = CONVEX
    """
    @staticmethod
    def sign_mul(sign, lh_size, curv, rh_size):
        cvx_mat = bu.mul(sign.pos_mat, lh_size, 
                              curv.cvx_mat, rh_size) | \
                  bu.mul(sign.neg_mat, lh_size, 
                              curv.conc_mat, rh_size)
        conc_mat = bu.mul(sign.pos_mat, lh_size,
                               curv.conc_mat, rh_size) | \
                   bu.mul(sign.neg_mat, lh_size,
                               curv.cvx_mat, rh_size)
        return Curvature(cvx_mat, conc_mat, curv.constant)

    # Equivalent to NEGATIVE * self
    def __neg__(self):
        return Curvature(self.conc_mat, self.cvx_mat, self.constant)

    # Comparison.
    def __eq__(self, other):
        return self.cvx_mat == other.cvx_mat and self.conc_mat == other.conc_mat

    # Promotes cvx_mat and conc_mat to BoolMats of the given size.
    def promote(self, size):
        cvx_mat = BoolMat.promote(self.cvx_mat, size)
        conc_mat = BoolMat.promote(self.conc_mat, size)
        return Curvature(cvx_mat, conc_mat, self.constant)

    # To string methods.
    def __repr__(self):
        return "Curvature(%s, %s)" % (self.cvx_mat, self.conc_mat)

    def __str__(self):
        return "negative entries = %s, positive entries = %s" % \
            (self.cvx_mat, self.conc_mat)

# Scalar signs.
Curvature.CONVEX = Curvature.name_to_curvature(Curvature.CONVEX_KEY)
Curvature.CONCAVE = Curvature.name_to_curvature(Curvature.CONCAVE_KEY)
Curvature.UNKNOWN = Curvature.name_to_curvature(Curvature.UNKNOWN_KEY)
Curvature.AFFINE = Curvature.name_to_curvature(Curvature.AFFINE_KEY)
Curvature.CONSTANT = Curvature.name_to_curvature(Curvature.CONSTANT_KEY)