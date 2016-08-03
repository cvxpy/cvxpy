"""
Copyright 2013 Steven Diamond and Xinyue Shen.

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

from cvxpy.atoms import reshape, vec
from cvxpy.expressions.constants import Constant


def linearize(expr):
    """Returns the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

    Returns None if cannot be linearized.

    Args:
        expr: An expression.

    Returns:
        An affine expression or None.
    """
    expr = Constant.cast_to_const(expr)
    if expr.is_affine():
        return expr
    else:
        tangent = expr.value
        if tangent is None:
            raise ValueError(
                "Cannot linearize non-affine expression with missing variable values."
            )
        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            elif var.is_matrix():
                flattened = Constant(grad_map[var]).T*vec(var - var.value)
                tangent = tangent + reshape(flattened, *expr.size)
            else:
                tangent = tangent + Constant(grad_map[var]).T*(var - var.value)
        return tangent
