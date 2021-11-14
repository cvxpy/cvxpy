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
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.expressions.constants.parameter import Parameter


# Sublevel sets for quasiconvex atoms.
#
# In the below functions, False is a placeholder for an infeasible
# constraint (one that cannot be represented in a DCP way), and True
# is a placeholder for the absence of a constraint
def dist_ratio_sub(expr, t):
    x = expr.args[0]
    a = expr.a
    b = expr.b

    def sublevel_set():
        if t.value > 1:
            return False
        tsq = t.value**2
        return ((1-tsq**2)*atoms.sum_squares(x) -
                atoms.matmul(2*(a-tsq*b), x) + atoms.sum_squares(a) -
                tsq*atoms.sum_squares(b)) <= 0
    return [sublevel_set]


def mul_sup(expr, t):
    x, y = expr.args
    if x.is_nonneg() and y.is_nonneg():
        return [x >= t * atoms.inv_pos(y)]
    elif x.is_nonpos() and y.is_nonpos():
        return [-x >= t * atoms.inv_pos(-y)]
    else:
        raise ValueError("Incorrect signs.")


def mul_sub(expr, t):
    x, y = expr.args
    if x.is_nonneg() and y.is_nonpos():
        return [y <= t * atoms.inv_pos(x)]
    elif x.is_nonpos() and y.is_nonneg():
        return [x <= t * atoms.inv_pos(y)]
    else:
        raise ValueError("Incorrect signs.")


def ratio_sup(expr, t):
    x, y = expr.args
    if y.is_nonneg():
        return [x >= t * y]
    elif y.is_nonpos():
        return [x <= t * y]
    else:
        raise ValueError("The denominator's sign must be known.")


def ratio_sub(expr, t):
    x, y = expr.args
    if y.is_nonneg():
        return [x <= t * y]
    elif y.is_nonpos():
        return [x >= t * y]
    else:
        raise ValueError("The denominator's sign must be known.")


def length_sub(expr, t):
    arg = expr.args[0]
    if isinstance(t, Parameter):
        def sublevel_set():
            if t.value < 0:
                return False
            if t.value >= arg.size:
                return True
            return arg[int(atoms.floor(t).value):] == 0
        return [sublevel_set]
    else:
        return [arg[int(atoms.floor(t).value):] == 0]


def sign_sup(expr, t):
    x = expr.args[0]

    def superlevel_set():
        if t.value <= -1:
            return True
        elif t.value <= 1:
            return x >= 0
        else:
            return False
    return [superlevel_set]


def sign_sub(expr, t):
    x = expr.args[0]

    def sublevel_set():
        if t.value >= 1:
            return True
        elif t.value >= -1:
            return x <= 0
        else:
            return False
    return [sublevel_set]


def gen_lambda_max_sub(expr, t):
    return [expr.args[0] == expr.args[0].T,
            expr.args[1] >> 0,
            (t * expr.args[1] - expr.args[0] >> 0)]


SUBLEVEL_SETS = {
    atoms.multiply: mul_sub,
    bin_op.DivExpression: ratio_sub,
    atoms.length: length_sub,
    atoms.sign: sign_sub,
    atoms.dist_ratio: dist_ratio_sub,
    atoms.gen_lambda_max: gen_lambda_max_sub,
}


SUPERLEVEL_SETS = {
    atoms.multiply: mul_sup,
    bin_op.DivExpression: ratio_sup,
    atoms.sign: sign_sup,
}


def sublevel(expr, t):
    """Return the t-level sublevel set for `expr`.

    Returned as a constraint phi_t(x) <= 0, where phi_t(x) is convex.
    """
    try:
        return SUBLEVEL_SETS[type(expr)](expr, t)
    except KeyError:
        raise RuntimeError(
                f"The {type(expr)} atom is not yet supported in DQCP. Please "
                "file an issue here: https://github.com/cvxpy/cvxpy/issues")


def superlevel(expr, t):
    """Return the t-level superlevel set for `expr`.

    Returned as a constraint phi_t(x) >= 0, where phi_t(x) is concave.
    """
    try:
        return SUPERLEVEL_SETS[type(expr)](expr, t)
    except KeyError:
        raise RuntimeError(
                f"The {type(expr)} atom is not yet supported in DQCP. Please "
                "file an issue here: https://github.com/cvxpy/cvxpy/issues")
