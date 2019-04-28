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
from cvxpy.atoms import inv_pos, multiply


def mul_sup(expr, t):
    x, y = expr.args
    if x.is_nonneg() and y.is_nonneg():
        return [x >= t * inv_pos(y)]
    elif x.is_nonpos() and y.is_nonpos():
        return [-x >= t * inv_pos(-y)]
    else:
        raise ValueError("Incorrect signs.")


def mul_sub(expr, t):
    x, y = expr.args
    if x.is_nonneg() and y.is_nonpos():
        return [y <= t * inv_pos(x)]
    elif x.is_nonpos() and y.is_nonneg():
        return [x <= t * inv_pos(y)]
    else:
        raise ValueError("Incorrect signs.")


SUBLEVEL_SETS = {
    multiply: mul_sub,
}


SUPERLEVEL_SETS = {
    multiply: mul_sup,
}


def sublevel(expr, t):
    """Return the t-level sublevel set for `expr`.

    Returned as a constraint phi_t(x) <= 0, where phi_t(x) is convex.
    """
    return SUBLEVEL_SETS[type(expr)](expr, t)


def superlevel(expr, t):
    """Return the t-level superlevel set for `expr`.

    Returned as a constraint phi_t(x) >= 0, where phi_t(x) is concave.
    """
    return SUPERLEVEL_SETS[type(expr)](expr, t)
