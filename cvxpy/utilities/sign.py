from typing import Tuple

"""
Copyright 2013 Steven Diamond

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


def sum_signs(exprs) -> Tuple[bool, bool]:
    """Give the sign resulting from summing a list of expressions.

    Args:
        shapes: A list of sign (is pos, is neg) tuples.

    Returns:
        The sign (is pos, is neg) of the sum.
    """
    is_pos = all(expr.is_nonneg() for expr in exprs)
    is_neg = all(expr.is_nonpos() for expr in exprs)
    return (is_pos, is_neg)


def mul_sign(lh_expr, rh_expr) -> Tuple[bool, bool]:
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
