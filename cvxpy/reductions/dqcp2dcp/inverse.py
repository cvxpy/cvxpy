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
from cvxpy import atoms
from cvxpy.atoms.affine.binary_operators import DivExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
import numpy as np

INVERTIBLE = set(
    [atoms.ceil, atoms.floor, NegExpression, atoms.exp, atoms.log, atoms.log1p,
     atoms.logistic, atoms.power])


# Inverses are extended-value functions
def inverse(expr):
    if type(expr) == atoms.ceil:
        return lambda t: atoms.floor(t)
    elif type(expr) == atoms.floor:
        return lambda t: atoms.ceil(t)
    elif type(expr) == NegExpression:
        return lambda t: -t
    elif type(expr) == atoms.exp:
        return lambda t: atoms.log(t) if t.value > 0 else -np.inf
    elif type(expr) == atoms.log:
        return lambda t: atoms.exp(t)
    elif type(expr) == atoms.log1p:
        return lambda t: atoms.exp(t) - 1
    elif type(expr) == atoms.logistic:
        return lambda t: atoms.log(atoms.exp(t) - 1) if t.value > 0 else -np.inf
    elif type(expr) == atoms.power:
        def power_inv(t):
            if expr.p == 1:
                return t
            return atoms.power(t, 1/expr.p) if t.value > 0 else np.inf
        return power_inv
    elif type(expr) == atoms.multiply:
        if expr.args[0].is_constant():
            const = expr.args[0]
        else:
            const = expr.args[1]
        return lambda t: t / const
    elif type(expr) == DivExpression:
        if expr.args[0].is_constant():
            const = expr.args[0]
        else:
            const = expr.args[1]
        return lambda t: t * const
    else:
        raise ValueError


def invertible(expr):
    if isinstance(expr, atoms.multiply) or isinstance(expr, DivExpression):
        return len(expr._non_const_idx()) == 1
    return type(expr) in INVERTIBLE
