"""
Copyright 2013 Steven Diamond and Xinyue Shen.

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
