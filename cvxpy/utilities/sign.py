"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
