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

from curvature import Curvature

class Monotonicity(object):
    """ Monotonicity of atomic functions in a given argument. """
    INCREASING_KEY = 'INCREASING'
    DECREASING_KEY = 'DECREASING'
    SIGNED_KEY = 'SIGNED'
    NONMONOTONIC_KEY = 'NONMONOTONIC'

    MONOTONICITY_SET = set([
        INCREASING_KEY, 
        DECREASING_KEY,
        SIGNED_KEY,
        NONMONOTONIC_KEY,
    ])

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
    def dcp_curvature(self, func_curvature, arg_sign, arg_curvature):
        if self.monotonicity_str == Monotonicity.INCREASING_KEY:
            return func_curvature + arg_curvature
        elif self.monotonicity_str == Monotonicity.DECREASING_KEY:
            return func_curvature - arg_curvature
        # Absolute value style monotonicity.
        elif self.monotonicity_str == Monotonicity.SIGNED_KEY:
            return func_curvature # TODO cvx + neg & cvx + pos & conc
        else: # non-monotonic
            return func_curvature + arg_curvature - arg_curvature

# Class constants for all monotonicity types.
Monotonicity.INCREASING = Monotonicity(Monotonicity.INCREASING_KEY)
Monotonicity.DECREASING = Monotonicity(Monotonicity.DECREASING_KEY)
Monotonicity.SIGNED = Monotonicity(Monotonicity.SIGNED_KEY)
Monotonicity.NONMONOTONIC = Monotonicity(Monotonicity.NONMONOTONIC_KEY)