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

class Curvature(object):
    """ Curvature for a convex optimization expression. """
    CONSTANT_KEY = 'CONSTANT'
    AFFINE_KEY = 'AFFINE'
    CONVEX_KEY = 'CONVEX'
    CONCAVE_KEY = 'CONCAVE'
    UNKNOWN_KEY = 'UNKNOWN'
    
    # List of valid curvature strings.
    CURVATURE_STRINGS = [CONSTANT_KEY, AFFINE_KEY, CONVEX_KEY, 
                       CONCAVE_KEY, UNKNOWN_KEY]
    # For multiplying curvature by negative sign.
    NEGATION_MAP = {CONVEX_KEY: CONCAVE_KEY, CONCAVE_KEY: CONVEX_KEY}
    
    def __init__(self,curvature_str):
        curvature_str = curvature_str.upper()
        if curvature_str in Curvature.CURVATURE_STRINGS:
            self.curvature_str = curvature_str
        else:
            raise Exception("No such curvature %s exists." % str(curvature_str))
        
    def __repr__(self):
        return "Curvature('%s')" % self.curvature_str
    
    def __str__(self):
        return self.curvature_str

    # Returns whether the curvature is constant.
    def is_constant(self):
        return self == Curvature.CONSTANT

    # Returns whether the curvature is affine, 
    # counting constant expressions as affine.
    def is_affine(self):
        return self.is_constant() or self == Curvature.AFFINE

    # Returns whether the curvature is convex, 
    # counting affine and constant expressions as convex.
    def is_convex(self):
        return self.is_affine() or self == Curvature.CONVEX

    # Returns whether the curvature is concave, 
    # counting affine and constant expressions as concave.
    def is_concave(self):
        return self.is_affine() or self == Curvature.CONCAVE

    # Returns whether the curvature is unknown.
    def is_unknown(self):
        return self == Curvature.UNKNOWN

    # Sums list of curvatures
    @staticmethod
    def sum(curvatures):
        return reduce(lambda x,y: x+y, curvatures)

    """
    Resolves the logic of adding curvatures.
      CONSTANT + ANYTHING = ANYTHING
      AFFINE + NONCONSTANT = NONCONSTANT
      CONVEX + CONCAVE = UNKNOWN
      SAME + SAME = SAME
    """
    def __add__(self, other):
        if self.is_constant():
            return other
        elif self.is_affine() and other.is_affine():
            return Curvature.AFFINE
        elif self.is_convex() and other.is_convex():
            return Curvature.CONVEX
        elif self.is_concave() and other.is_concave():
            return Curvature.CONCAVE
        else:
            return Curvature.UNKNOWN
    
    def __sub__(self, other):
        return self + -other
       
    def __mul__(self, other):
        if self == Curvature.CONSTANT or other == Curvature.CONSTANT:
            return self + other
        else:
            return Curvature.UNKNOWN
        
    def __neg__(self):
        curvature_str = Curvature.NEGATION_MAP.get(self.curvature_str, 
                                                   self.curvature_str)
        return Curvature(curvature_str)
        
    def __eq__(self,other):
        return self.curvature_str == other.curvature_str
    
    def __ne__(self,other):
        return self.curvature_str != other.curvature_str

# Class constants for all curvature types.
Curvature.CONSTANT = Curvature(Curvature.CONSTANT_KEY)
Curvature.AFFINE = Curvature(Curvature.AFFINE_KEY)
Curvature.CONVEX = Curvature(Curvature.CONVEX_KEY)
Curvature.CONCAVE = Curvature(Curvature.CONCAVE_KEY)
Curvature.UNKNOWN = Curvature(Curvature.UNKNOWN_KEY)
Curvature.NONCONVEX = Curvature(Curvature.UNKNOWN_KEY)