class Curvature(object):
    """ Curvature for a convex optimization expression. """
    CONSTANT_KEY = 'CONSTANT'
    AFFINE_KEY = 'AFFINE'
    CONVEX_KEY = 'CONVEX'
    CONCAVE_KEY = 'CONCAVE'
    UNKNOWN_KEY = 'UNKNOWN'
    
    """
    VEXITY_MAP for resolving curvature addition using bitwise OR:
      CONSTANT (0) | ANYTHING = ANYTHING
      AFFINE (1) | NONCONSTANT = NONCONSTANT
      CONVEX (3) | CONCAVE (5) = UNKNOWN (7)
      SAME | SAME = SAME
    """
    VEXITY_MAP = {
                  CONSTANT_KEY: 0,
                  AFFINE_KEY: 1, 
                  CONVEX_KEY: 3, 
                  CONCAVE_KEY: 5,
                  UNKNOWN_KEY: 7
                 }

    INVERSE_VEXITY_MAP = dict( (v,k) for k,v in VEXITY_MAP.items() )

    # For multiplying curvature by negative sign.
    NEGATION_MAP = {CONVEX_KEY: CONCAVE_KEY, CONCAVE_KEY: CONVEX_KEY}
    
    def __init__(self,curvature_str):
        curvature_str = curvature_str.upper()
        if curvature_str in Curvature.VEXITY_MAP.keys():
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

    # Sums list of curvatures
    @staticmethod
    def sum(curvatures):
        sum_curvature = Curvature.CONSTANT
        for curvature in curvatures:
            sum_curvature = sum_curvature + curvature
        return sum_curvature

    def __add__(self, other):
        curvature_val = Curvature.VEXITY_MAP[self.curvature_str] | \
                        Curvature.VEXITY_MAP[other.curvature_str]
        return Curvature(Curvature.INVERSE_VEXITY_MAP[curvature_val])
    
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