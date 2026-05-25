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

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.power_canon import power_approx_canon
from cvxpy.reductions.dcp2cone.canonicalizers.quad_over_lin_canon import quad_over_lin_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.solver_context import SolverInfo


def huber_canon(expr, args, solver_context: SolverInfo | None = None):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(shape)
    s = Variable(shape)

    # n**2 + 2*M*|s|
    # TODO(akshayka): Make use of recursion inherent to canonicalization
    # process and just return a power / abs expressions for readability sake
    power_expr = power(n, 2)
    n2, constr_sq = power_approx_canon(power_expr, power_expr.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)
    obj = n2 + 2 * M * abs_s

    # x == s + n
    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)
    return obj, constraints

def huber_perspective_canon(expr, args: list, solver_context: SolverInfo | None = None):
    """Canonicalize the three-argument perspective Huber atom.

    Uses the reparametrized splitting: let ñ = t*n and s̃ = t*s, so that
    the bilinear equality x == t*s + t*n becomes affine: x == s̃ + ñ.
    Then:

        t * n^2     = ñ^2 / t  = quad_over_lin(ñ, t)
        t * |s|     = |s̃|      = abs(s̃)

    giving:

        t * huber(x/t, M) = min_{ñ, s̃}  quad_over_lin(ñ, t) + 2*M*|s̃|
        subject to:  x == s̃ + ñ
                     t >= 0

    All constraints are now affine, and the two penalty terms recurse into
    canonicalizers CVXPY already has (quad_over_lin and abs), exactly
    mirroring the structure of the 2-arg canon.
    """
    M = expr.M
    x = args[0]    # canonicalized x, affine
    t = args[1]    # canonicalized t, scalar, concave/affine
    shape = x.shape

    # Reparametrized auxiliaries: ñ = t*n,  s̃ = t*s
    n_tilde = Variable(shape, name="huber_persp_n")
    s_tilde = Variable(shape, name="huber_persp_s")

    if shape:
        size = n_tilde.size
        n_col = n_tilde.flatten(order='F').reshape((size, 1), order='F')
        qol_expr = quad_over_lin(n_col, t, axis=1)
        qol_obj_flat, constr_qol = quad_over_lin_canon(qol_expr, qol_expr.args)
        # qol_obj_flat is shape (size,); un-flatten for 2-D inputs
        qol_obj = qol_obj_flat.reshape(shape, order='F') if len(shape) > 1 else qol_obj_flat
    else:
        # Scalar x: a single quad_over_lin is exact.
        qol_expr = quad_over_lin(n_tilde, t)
        qol_obj, constr_qol = quad_over_lin_canon(qol_expr, qol_expr.args)

    # t * |s| --> abs(s_tilde)
    abs_expr = abs(s_tilde)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)

    obj = qol_obj + 2 * M * abs_s

    constraints = constr_qol + constr_abs
    constraints.append(x == s_tilde + n_tilde)
    constraints.append(t >= 0)

    return obj, constraints