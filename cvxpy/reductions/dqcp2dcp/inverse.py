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
from cvxpy.atoms import ceil, exp, floor, inv_pos, log, log1p, logistic, power
from cvxpy.atoms.affine.unary_operators import NegExpression

INVERTIBLE = set(
    [ceil, floor, NegExpression, exp, log, log1p, logistic, power, inv_pos])


def inverse(expr):
    if type(expr) == ceil:
        return lambda t: floor(t)
    elif type(expr) == floor:
        return lambda t: ceil(t)
    elif type(expr) == NegExpression:
        return lambda t: -t
    elif type(expr) == exp:
        return lambda t: log(t)
    elif type(expr) == log:
        return lambda t: exp(t)
    elif type(expr) == log1p:
        return lambda t: exp(t) - 1
    elif type(expr) == logistic:
        return lambda t: log(exp(t) - 1)
    elif type(expr) == power:
        return lambda t: power(t, 1/expr.p),
    elif type(expr) == inv_pos:
        return lambda t: inv_pos(t)
    else:
        raise ValueError


def invertible(expr):
    return type(expr) in INVERTIBLE
