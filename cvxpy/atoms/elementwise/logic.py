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

from functools import reduce
from typing import List, Tuple

import numpy as np

from cvxpy.atoms.elementwise.elementwise import Elementwise


def _is_boolean_arg(arg):
    """Check if an argument is a valid boolean logic input."""
    if isinstance(arg, LogicExpression):
        return True
    from cvxpy.expressions.leaf import Leaf
    if isinstance(arg, Leaf) and arg.attributes.get('boolean'):
        return True
    from cvxpy.expressions.constants.constant import Constant
    if isinstance(arg, Constant) and arg.is_boolean_valued:
        return True
    return False


class LogicExpression(Elementwise):
    """Base class for boolean logic atoms (Not, And, Or, Xor)."""

    def validate_arguments(self) -> None:
        super().validate_arguments()
        for arg in self.args:
            if not _is_boolean_arg(arg):
                raise ValueError(
                    f"All arguments to {self.__class__.__name__} must be "
                    f"boolean variables or LogicExpression instances. "
                    f"Got {type(arg).__name__}."
                )

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Result is boolean (0 or 1), so nonneg."""
        return (True, False)

    def is_atom_convex(self) -> bool:
        return True

    def is_atom_concave(self) -> bool:
        return True

    def is_incr(self, idx) -> bool:
        return False

    def is_decr(self, idx) -> bool:
        return False

    def _grad(self, values) -> List:
        return [None for _ in values]


class Not(LogicExpression):
    """Logical NOT of a boolean expression.

    Returns ``1 - x``, i.e., flips 0 to 1 and 1 to 0.

    Can also be written with the ``~`` operator: ``~x``.

    Parameters
    ----------
    arg : Expression
        A boolean variable or LogicExpression.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        not_x = ~x                # operator syntax
        not_x = cp.logic.Not(x)   # equivalent functional syntax
    """

    def __init__(self, arg) -> None:
        super().__init__(arg)

    def validate_arguments(self) -> None:
        if len(self.args) != 1:
            raise TypeError("Not takes exactly 1 argument.")
        super().validate_arguments()

    def is_decr(self, idx) -> bool:
        return True

    def name(self) -> str:
        child = self.args[0]
        if isinstance(child, _NaryLogicExpression):
            return "~(" + child.name() + ")"
        return "~" + child.name()

    def format_labeled(self):
        if self._label is not None:
            return self._label
        child = self.args[0]
        if isinstance(child, _NaryLogicExpression):
            return "~(" + child.format_labeled() + ")"
        return "~" + child.format_labeled()

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return 1 - values[0]


class _NaryLogicExpression(LogicExpression):
    """Shared base for n-ary logic atoms (And, Or, Xor).

    Subclasses set ``OP_NAME`` and ``_PAREN_TYPES`` class attributes;
    formatting methods are inherited from here.
    """

    OP_NAME: str
    _PAREN_TYPES: Tuple[str, ...]

    def __init__(self, arg1, arg2, *args) -> None:
        super().__init__(arg1, arg2, *args)

    def _format_child(self, child, use_labels: bool = False) -> str:
        text = child.format_labeled() if use_labels else child.name()
        if isinstance(child, LogicExpression) and \
                type(child).__name__ in self._PAREN_TYPES:
            return "(" + text + ")"
        return text

    def name(self) -> str:
        return self.OP_NAME.join(
            self._format_child(a) for a in self.args
        )

    def format_labeled(self):
        if self._label is not None:
            return self._label
        return self.OP_NAME.join(
            self._format_child(a, use_labels=True) for a in self.args
        )


class And(_NaryLogicExpression):
    """Logical AND of boolean expressions.

    Returns 1 if and only if all arguments equal 1, and 0 otherwise.

    For two operands, can also be written with the ``&`` operator: ``x & y``.

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        both = x & y                  # operator syntax
        both = cp.logic.And(x, y)     # equivalent functional syntax
        all3 = cp.logic.And(x, y, z)  # n-ary (3+ args) requires functional syntax
    """

    OP_NAME = " & "
    # Or and Xor have lower precedence than &, so parenthesize them.
    _PAREN_TYPES = ("Or", "Xor")

    def is_incr(self, idx) -> bool:
        return True

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(np.minimum, values)


class Or(_NaryLogicExpression):
    """Logical OR of boolean expressions.

    Returns 1 if and only if at least one argument equals 1, and 0 otherwise.

    For two operands, can also be written with the ``|`` operator: ``x | y``.

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        either = x | y                # operator syntax
        either = cp.logic.Or(x, y)    # equivalent functional syntax
        any3 = cp.logic.Or(x, y, z)   # n-ary (3+ args) requires functional syntax
    """

    OP_NAME = " | "
    # | has lowest precedence among logic ops; no children need parens.
    _PAREN_TYPES = ()

    def is_incr(self, idx) -> bool:
        return True

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(np.maximum, values)


class Xor(_NaryLogicExpression):
    """Logical XOR of boolean expressions.

    For two arguments: result is 1 iff exactly one is 1.
    For n arguments: result is 1 iff an odd number are 1 (parity).

    For two operands, can also be written with the ``^`` operator: ``x ^ y``.

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        exclusive = x ^ y                 # operator syntax
        exclusive = cp.logic.Xor(x, y)    # equivalent functional syntax
        parity3 = cp.logic.Xor(x, y, z)   # n-ary (3+ args) requires functional syntax
    """

    OP_NAME = " ^ "
    # Or has lower precedence than ^, so parenthesize it.
    _PAREN_TYPES = ("Or",)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(lambda a, b: np.mod(a + b, 2), values)


def implies(x, y):
    """Logical implication: x => y.

    Returns 1 unless x = 1 and y = 0.  Equivalent to ``Or(Not(x), y)``.

    Parameters
    ----------
    x : Expression
        A boolean variable or LogicExpression.
    y : Expression
        A boolean variable or LogicExpression.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = cp.logic.implies(x, y)
    """
    return Or(Not(x), y)


def iff(x, y):
    """Logical biconditional: x <=> y.

    Returns 1 if and only if x and y have the same value.
    Equivalent to ``Not(Xor(x, y))``.

    Parameters
    ----------
    x : Expression
        A boolean variable or LogicExpression.
    y : Expression
        A boolean variable or LogicExpression.

    Examples
    --------

    .. code-block:: python

        import cvxpy as cp

        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = cp.logic.iff(x, y)
    """
    return Not(Xor(x, y))
