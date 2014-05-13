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

from cvxpy.utilities.curvature import Curvature

INCREASING = 'INCREASING'
DECREASING = 'DECREASING'
SIGNED = 'SIGNED'
NONMONOTONIC = 'NONMONOTONIC'

def dcp_curvature(monotonicity, func_curvature, arg_sign, arg_curvature):
    """Applies DCP composition rules to determine curvature in each argument.

    Composition rules:
        Key: Function curvature + monotonicity + argument curvature
             == curvature in argument
        anything + anything + constant == constant
        anything + anything + affine == original curvature
        convex/affine + increasing + convex == convex
        convex/affine + decreasing + concave == convex
        concave/affine + increasing + concave == concave
        concave/affine + decreasing + convex == concave
    Notes: Increasing (decreasing) means non-decreasing (non-increasing).
           Any combinations not covered by the rules result in a
           nonconvex expression.

    Args:
        monotonicity: The monotonicity of the function in the given argument.
        func_curvature: The curvature of the function.
        arg_sign: The sign of the given argument.
        arg_curvature: The curvature of the given argument.

    Returns:
        The Curvature of the composition of function and arguments.
    """
    if arg_curvature.is_constant():
        result_curv = Curvature.CONSTANT
    elif arg_curvature.is_affine():
        result_curv = func_curvature
    elif monotonicity == INCREASING:
        result_curv = func_curvature + arg_curvature
    elif monotonicity == DECREASING:
        result_curv = func_curvature - arg_curvature
    # Absolute value style monotonicity.
    elif monotonicity == SIGNED and \
         func_curvature.is_convex():
        if (arg_curvature.is_convex() and arg_sign.is_positive()) or \
           (arg_curvature.is_concave() and arg_sign.is_negative()):
            result_curv = func_curvature
        else:
            result_curv = Curvature.UNKNOWN
    else: # non-monotonic
        result_curv = func_curvature + arg_curvature - arg_curvature

    return result_curv
