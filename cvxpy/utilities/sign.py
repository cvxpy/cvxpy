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
    is_pos = all(expr.is_nonneg() for expr in exprs)
    is_neg = all(expr.is_nonpos() for expr in exprs)
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

    lh_nonneg = lh_expr.is_nonneg()
    rh_nonneg = rh_expr.is_nonneg()
    lh_nonpos = lh_expr.is_nonpos()
    rh_nonpos = rh_expr.is_nonpos()

    lh_zero = lh_nonneg and lh_nonpos
    rh_zero = rh_nonneg and rh_nonpos

    is_zero = lh_zero or rh_zero

    is_pos = is_zero or (lh_nonneg and rh_nonneg) or (lh_nonpos and rh_nonpos)
    is_neg = is_zero or (lh_nonneg and rh_nonpos) or (lh_nonpos and rh_nonneg)
    return (is_pos, is_neg)
