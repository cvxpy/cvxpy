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
import numpy as np

from cvxpy import atoms
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.unary_operators import NegExpression

# these atoms are always invertible. others (like AddExpression, DivExpression,
# Sum, and cumsum) are only invertible in special cases, checked in the
# `invertible` function.
INVERTIBLE = set(
    [atoms.ceil, atoms.floor, NegExpression, atoms.exp, atoms.log, atoms.log1p,
     atoms.logistic, atoms.power, atoms.abs])


# Inverses are extended-value functions
def inverse(expr):
    if type(expr) == atoms.ceil:
        return lambda t: atoms.floor(t)
    elif type(expr) == atoms.floor:
        return lambda t: atoms.ceil(t)
    elif type(expr) == NegExpression:
        return lambda t: -t
    elif type(expr) == atoms.exp:
        return lambda t: atoms.log(t) if t.is_nonneg() else -np.inf
    elif type(expr) == atoms.log:
        return lambda t: atoms.exp(t)
    elif type(expr) == atoms.log1p:
        return lambda t: atoms.exp(t) - 1
    elif type(expr) == atoms.logistic:
        return lambda t: atoms.log(atoms.exp(t) - 1) if t.is_nonneg() else -np.inf
    elif type(expr) == atoms.power:
        def power_inv(t):
            if expr.p.value == 1:
                return t
            return atoms.power(t, 1/expr.p.value) if t.is_nonneg() else np.inf
        return power_inv
    elif type(expr) == atoms.multiply:
        if expr.args[0].is_constant():
            const = expr.args[0]
        else:
            const = expr.args[1]
        return lambda t: t / const
    elif type(expr) == DivExpression:
        # either const / x <= t or x / const <= t
        if expr.args[0].is_constant():
            # numerator is constant
            const = expr.args[0]
            return lambda t: const / t
        else:
            # denominator is constant
            const = expr.args[1]
            return lambda t: const * t
    elif type(expr) == AddExpression:
        if expr.args[0].is_constant():
            const = expr.args[0]
        else:
            const = expr.args[1]
        return lambda t: t - const
    elif type(expr) == atoms.abs:
        arg = expr.args[0]
        if arg.is_nonneg():
            return lambda t: t
        elif arg.is_nonpos():
            return lambda t: -t
        else:
            raise ValueError("Sign of argument must be known.")
    elif type(expr) in (Sum, atoms.cumsum):
        return lambda t: t
    else:
        raise ValueError


def invertible(expr):
    if (isinstance(expr, atoms.multiply) or isinstance(expr, DivExpression) or
            isinstance(expr, AddExpression)):
        return len(expr._non_const_idx()) == 1
    elif isinstance(expr, (Sum, atoms.cumsum)):
        return expr._is_real()
    else:
        return type(expr) in INVERTIBLE
