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

from cvxpy import settings
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


def _pnorm_p2_canon(expr, args):
    """Handle p == 2 case via SOC directly (shared by exact and approx)."""
    x = args[0]
    axis = expr.axis
    shape = expr.shape
    t = Variable(shape)
    if axis is None:
        assert shape == tuple()
        return t, [SOC(t, vec(x, order="F"))]
    else:
        return t, [SOC(vec(t, order="F"), x, axis)]


def pnorm_exact_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize Pnorm using power cone constraints."""
    p = expr.p

    if p == 2:
        return _pnorm_p2_canon(expr, args)

    x = args[0]
    shape = expr.shape
    t = Variable(shape)

    constraints = []
    if p > 1:
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    r = Variable(x.shape)
    constraints += [sum(r) == t]

    promoted_t = Constant(np.ones(x.shape)) * t

    if p < 0:
        alpha = float(-p / (1 - p))
        constraints += [
            PowCone3D(vec(x, order="F"), vec(r, order="F"), vec(promoted_t, order="F"), alpha)
        ]
    elif 0 < p < 1:
        alpha = float(p)
        constraints += [
            PowCone3D(vec(x, order="F"), vec(promoted_t, order="F"), vec(r, order="F"), alpha)
        ]
    elif p > 1:
        alpha = float(1 / p)
        constraints += [
            PowCone3D(vec(r, order="F"), vec(promoted_t, order="F"), vec(x, order="F"), alpha)
        ]

    return t, constraints


def pnorm_approx_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize PnormApprox using SOC constraints via rational approximation."""
    p = expr.p

    if p == 2:
        return _pnorm_p2_canon(expr, args)

    x = args[0]
    p = Fraction(p)
    shape = expr.shape
    t = Variable(shape)

    constraints = []
    if p > 1:
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

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
