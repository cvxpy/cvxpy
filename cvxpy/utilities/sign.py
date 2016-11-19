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


def sum_signs(exprs):
    """Give the sign resulting from summing a list of expressions.

    Args:
        shapes: A list of sign (is pos, is neg) tuples.

    Returns:
        The sign (is pos, is neg) of the sum.
    """
    is_pos = all([expr.is_positive() for expr in exprs])
    is_neg = all([expr.is_negative() for expr in exprs])
    return (is_pos, is_neg)


def mul_sign(lh_expr, rh_expr):
    """Give the sign resulting from multiplying two expressions.

    Args:
        lh_expr: An expression.
        rh_expr: An expression.

    Returns:
        The sign (is pos, is neg) of the product.
    """
    # ZERO * ANYTHING == ZERO
    # POSITIVE * POSITIVE == POSITIVE
    # NEGATIVE * POSITIVE == NEGATIVE
    # NEGATIVE * NEGATIVE == POSITIVE
    is_pos = (lh_expr.is_zero() or rh_expr.is_zero()) or \
             (lh_expr.is_positive() and rh_expr.is_positive()) or \
             (lh_expr.is_negative() and rh_expr.is_negative())
    is_neg = (lh_expr.is_zero() or rh_expr.is_zero()) or \
             (lh_expr.is_positive() and rh_expr.is_negative()) or \
             (lh_expr.is_negative() and rh_expr.is_positive())
    return (is_pos, is_neg)
