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

    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    def is_incr(self, idx) -> bool:
        return False

    def is_decr(self, idx) -> bool:
        return False

    def _grad(self, values) -> List:
        return [None for _ in values]


class Not(LogicExpression):
    """Logical NOT of a boolean expression.

    Parameters
    ----------
    arg : Expression
        A boolean variable or LogicExpression.
    """

    def __init__(self, arg) -> None:
        super().__init__(arg)

    def validate_arguments(self) -> None:
        if len(self.args) != 1:
            raise TypeError("Not takes exactly 1 argument.")
        super().validate_arguments()

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return 1 - values[0]


class And(LogicExpression):
    """Logical AND of boolean expressions.

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        super().__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(np.minimum, values)


class Or(LogicExpression):
    """Logical OR of boolean expressions.

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        super().__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(np.maximum, values)


class Xor(LogicExpression):
    """Logical XOR of boolean expressions.

    For two arguments: result is 1 iff exactly one is 1.
    For n arguments: result is 1 iff an odd number are 1 (parity).

    Parameters
    ----------
    *args : Expression
        Two or more boolean variables or LogicExpressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        super().__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return reduce(lambda a, b: np.mod(a + b, 2), values)
