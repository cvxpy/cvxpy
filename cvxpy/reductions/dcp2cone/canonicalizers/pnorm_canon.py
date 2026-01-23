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

import warnings
from fractions import Fraction

import numpy as np

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.power_tools import gm_constrs
from cvxpy.utilities.solver_context import SolverInfo


def pnorm_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    p = expr.p
    axis = expr.axis
    shape = expr.shape
    t = Variable(shape)

    # p == 2 is a special case handled via SOC directly
    if p == 2:
        if axis is None:
            assert shape == tuple()
            return t, [SOC(t, vec(x, order="F"))]
        else:
            return t, [SOC(vec(t, order="F"), x, axis)]

    # If user requested approximation (default), use SOC
    if expr._approx:
        return pnorm_canon_approx(expr, args, solver_context)

    # User requested power cones (approx=False)
    # Check if solver supports them
    if solver_context is not None and PowCone3D in solver_context.solver_supported_constraints:
        return pnorm_canon_cone(expr, args)

    # Fallback to SOC if power cones not supported
    return pnorm_canon_approx(expr, args, solver_context=None)


def pnorm_canon_approx(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    p = Fraction(expr.p)
    shape = expr.shape
    t = Variable(shape)

    # we need an absolute value constraint for the symmetric convex branches
    # (p > 1)
    constraints = []
    if p > 1:
        # TODO(akshayka): Express this more naturally (recursively), in terms
        # of the other atoms
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    # now, we take care of the remaining convex and concave branches
    # to create the rational powers, we need a new variable, r, and
    # the constraint sum(r) == t
    r = Variable(x.shape)
    constraints += [sum(r) == t]

    promoted_t = Constant(np.ones(x.shape)) * t

    if p < 0:
        constraints += gm_constrs(promoted_t, [x, r], (-p / (1 - p), 1 / (1 - p)))
    elif 0 < p < 1:
        constraints += gm_constrs(r, [x, promoted_t], (p, 1 - p))
    elif p > 1:
        constraints += gm_constrs(x, [r, promoted_t], (1 / p, 1 - 1 / p))

    # Warn if the solver supports power cones and the approximation is poor
    solver_supports_powcone = (
        solver_context is not None and PowCone3D in solver_context.solver_supported_constraints
    )
    if solver_supports_powcone:
        approx_error = expr.approx_error
        num_soc = len([c for c in constraints if isinstance(c, SOC)])
        if (
            approx_error > settings.POWERCONE_APPROX_ERROR_THRESHOLD
            or num_soc > settings.POWERCONE_APPROX_SOC_THRESHOLD
        ):
            warnings.warn(
                f"pnorm with p={expr.original_p} is being approximated "
                f"(error: {approx_error:.2e}) using {num_soc} SOC constraints. "
                f"Consider using approx=False to use power cones instead.",
                stacklevel=6,
            )

    return t, constraints


def pnorm_canon_cone(expr, args):
    x = args[0]
    p = expr.p
    shape = expr.shape
    t = Variable(shape)

    # we need an absolute value constraint for the symmetric convex branches
    # (p > 1)
    constraints = []
    if p > 1:
        # TODO(akshayka): Express this more naturally (recursively), in terms
        # of the other atoms
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    # now, we take care of the remaining convex and concave branches
    # to create the rational powers, we need a new variable, r, and
    # the constraint sum(r) == t
    r = Variable(x.shape)
    constraints += [sum(r) == t]

    promoted_t = Constant(np.ones(x.shape)) * t

    # Use power cone constraints
    # PowCone3D: x^alpha * y^(1-alpha) >= |z|
    if p < 0:
        # promoted_t >= x^(-p/(1-p)) * r^(1/(1-p))
        # alpha = -p/(1-p), so x^alpha * r^(1-alpha) >= promoted_t
        alpha = float(-p / (1 - p))
        constraints += [
            PowCone3D(vec(x, order="F"), vec(r, order="F"), vec(promoted_t, order="F"), alpha)
        ]
    elif 0 < p < 1:
        # r >= x^p * promoted_t^(1-p)
        # alpha = p, so x^alpha * promoted_t^(1-alpha) >= r
        alpha = float(p)
        constraints += [
            PowCone3D(vec(x, order="F"), vec(promoted_t, order="F"), vec(r, order="F"), alpha)
        ]
    elif p > 1:
        # x >= r^(1/p) * promoted_t^(1-1/p)
        # alpha = 1/p, so r^alpha * promoted_t^(1-alpha) >= x
        alpha = float(1 / p)
        constraints += [
            PowCone3D(vec(r, order="F"), vec(promoted_t, order="F"), vec(x, order="F"), alpha)
        ]

    return t, constraints
