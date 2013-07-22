from cvxpy.expressions.curvature import Curvature

class Monotonicity(object):
    """ Monotonicity of atomic functions in a given argument. """
    INCREASING_KEY = 'INCREASING'
    DECREASING_KEY = 'DECREASING'
    NONMONOTONIC_KEY = 'NONMONOTONIC'

    MONOTONICITY_SET = set([INCREASING_KEY, DECREASING_KEY, NONMONOTONIC_KEY])

    def __init__(self,monotonicity_str):
        monotonicity_str = monotonicity_str.upper()
        if monotonicity_str in Monotonicity.MONOTONICITY_SET:
            self.monotonicity_str = monotonicity_str
        else:
            raise Exception("No such monotonicity %s exists." % str(monotonicity_str))

    def __repr__(self):
        return "Monotonicity('%s')" % self.monotonicity_str
    
    def __str__(self):
        return self.monotonicity_str
    
    """
    Applies DCP composition rules to determine curvature in each argument.
    Composition rules:
        Key: Function curvature + monotonicity + argument curvature == curvature in argument
        anything + anything + affine == original curvature
        convex/affine + increasing + convex == convex
        convex/affine + decreasing + concave == convex
        concave/affine + increasing + concave == concave
        concave/affine + decreasing + convex == concave
    Notes: Increasing (decreasing) means non-decreasing (non-increasing).
        Any combinations not covered by the rules result in a nonconvex expression.
    """
    def dcp_curvature(self, func_curvature, arg_curvature):
        if arg_curvature.is_affine():
            return func_curvature
        elif self.monotonicity_str == Monotonicity.INCREASING_KEY:
            return func_curvature + arg_curvature
        elif self.monotonicity_str == Monotonicity.DECREASING_KEY:
            return func_curvature - arg_curvature
        else: # non-monotonic
            return Curvature.UNKNOWN

# Class constants for all monotonicity types.
Monotonicity.INCREASING = Monotonicity(Monotonicity.INCREASING_KEY)
Monotonicity.DECREASING = Monotonicity(Monotonicity.DECREASING_KEY)
Monotonicity.NONMONOTONIC = Monotonicity(Monotonicity.NONMONOTONIC_KEY)