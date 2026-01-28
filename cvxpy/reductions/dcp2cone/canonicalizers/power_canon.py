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

import numpy as np

from cvxpy import settings
from cvxpy.constraints import PowCone3D
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs, powcone_constrs
from cvxpy.utilities.solver_context import SolverInfo


def power_exact_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize Power using power cone constraints."""
    x = args[0]
    p = expr.p_used

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []

    t = Variable(shape)

    if 0 < p < 1:
        alpha = float(p)
        return t, powcone_constrs(t, [x, ones], alpha)
    elif p > 1:
        alpha = float(1 / p)
        constrs = powcone_constrs(x, [t, ones], alpha)
        if p % 2 != 0:
            constrs += [x >= 0]
        return t, constrs
    elif p < 0:
        alpha = float(p / (p - 1))
        return t, powcone_constrs(ones, [x, t], alpha)
    else:
        raise NotImplementedError("This power is not yet supported.")


def power_approx_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize PowerApprox using SOC constraints via rational approximation."""
    x = args[0]
    p = expr.p_used
    w = expr.w

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []

    t = Variable(shape)
    if 0 < p < 1:
        constrs = gm_constrs(t, [x, ones], w)
    elif p > 1:
        constrs = gm_constrs(x, [t, ones], w)
    elif p < 0:
        constrs = gm_constrs(ones, [x, t], w)
    else:
        raise NotImplementedError("This power is not yet supported.")

    # Warn if the solver supports power cones and the approximation is poor
    solver_supports_powcone = (
        solver_context is not None and PowCone3D in solver_context.solver_supported_constraints
    )
    if solver_supports_powcone:
        approx_error = getattr(expr, "approx_error", 0.0)
        num_soc = len(constrs)
        if (
            approx_error > settings.POWERCONE_APPROX_ERROR_THRESHOLD
            or num_soc > settings.POWERCONE_APPROX_SOC_THRESHOLD
        ):
            warnings.warn(
                f"Power atom with exponent {float(expr._p_orig)} is being approximated "
                f"with rational {p} (error: {approx_error:.2e}) "
                f"using {num_soc} SOC constraints. "
                f"Consider using approx=False to use power cones instead.",
                stacklevel=6,
            )

    return t, constrs
