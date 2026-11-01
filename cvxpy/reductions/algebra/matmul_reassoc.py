"""
Copyright, the CVXPY authors

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
import operator
from functools import reduce

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.expressions.constants import Constant
from cvxpy.reductions.canonicalization import Canonicalization


def _is_plain_constant(expr):
    """Variable-free and parameter-free: a value that never changes between solves."""
    return expr.is_constant() and not expr.parameters()


def _is_vector(expr):
    """1-D, or 2-D with a singleton dimension."""
    return len(expr.shape) <= 1 or min(expr.shape) == 1


def _apply_constant_right(left, c):
    """Build ``left @ c`` for a plain-constant ``c``, pushing ``c`` toward the leaves:

      constant @ c   -> Constant(value)
      (A @ B) @ c    -> A @ (B @ c)
      (-E) @ c       -> -(E @ c)
      (E1 + E2) @ c  -> E1 @ c + E2 @ c   (vector c only)
    """
    if _is_plain_constant(left):
        return Constant(left.value @ c.value)
    if type(left) is MulExpression:
        a, b = left.args
        if _is_plain_constant(b):
            return _apply_constant_right(a, Constant(b.value @ c.value))
        if _is_vector(c):
            return a @ _apply_constant_right(b, c)
        if _is_plain_constant(a):
            return a @ (b @ c)
        return left @ c
    if type(left) is NegExpression:
        return -_apply_constant_right(left.args[0], c)
    if (type(left) is AddExpression and _is_vector(c)
            and all(arg.shape == left.shape for arg in left.args)):
        return reduce(operator.add, [_apply_constant_right(arg, c) for arg in left.args])
    return left @ c


def _normalize_matmul(expr):
    """Reassociate ``expr`` when a matmul chain ends in a plain-constant factor."""
    left, right = expr.args
    # A 1-D operand contracts to an inner product, and (a @ b) @ c != a @ (b @ c)
    # across that collapse; a 2-D left guarantees both its factors are 2-D.
    if len(left.shape) != 2:
        return expr
    if _is_plain_constant(right) and not left.is_constant():
        return _apply_constant_right(left, right)
    return expr


def _matmul_canon(expr, args):
    return _normalize_matmul(expr.copy(args)), []


class MatmulReassoc(Canonicalization):
    """Reassociate matmul chains that end in a constant, so that a chain like
    (A @ X) @ C folds its constants instead of building the wide A @ X (issue #2205)."""

    def __init__(self, problem=None) -> None:
        super().__init__(canon_methods={MulExpression: _matmul_canon}, problem=problem)

    def accepts(self, problem) -> bool:
        return True
