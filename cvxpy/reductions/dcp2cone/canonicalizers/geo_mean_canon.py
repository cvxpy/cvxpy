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
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
from cvxpy.utilities.solver_context import SolverInfo


def geo_mean_exact_canon(expr, args, solver_context: SolverInfo | None = None):
    # Exact path: produce PowConeND; the solving chain adds Exotic2Common
    # to decompose to PowCone3D when the solver lacks native PowConeND support.
    if solver_context is not None:
        supports_pow = (
            PowConeND in solver_context.solver_supported_constraints
            or PowCone3D in solver_context.solver_supported_constraints
        )
        if not supports_pow:
            raise ValueError(
                "GeoMean (exact) requires a solver that supports power cones, "
                "but the current solver supports neither PowConeND nor PowCone3D."
            )
    return _geo_mean_cone_canon(expr, args)


def geo_mean_approx_canon(expr, args, solver_context: SolverInfo | None = None):
    return _geo_mean_soc_canon(expr, args, solver_context)


def _geo_mean_soc_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    w = expr.w
    shape = expr.shape
    t = Variable(shape)

    if x.shape == ():
        x_list = [x]
    else:
        x_list = [x[i] for i in range(len(w))]

    constrs = gm_constrs(t, x_list, w)

    # Warn if the solver supports power cones and the approximation is poor
    solver_supports_powcone = (
        solver_context is not None and PowConeND in solver_context.solver_supported_constraints
    )
    if solver_supports_powcone:
        approx_error = expr.approx_error
        num_soc = len(constrs)
        if (
            approx_error > settings.POWERCONE_APPROX_ERROR_THRESHOLD
            or num_soc > settings.POWERCONE_APPROX_SOC_THRESHOLD
        ):
            warnings.warn(
                f"geo_mean is being approximated (error: {approx_error:.2e}) "
                f"using {num_soc} SOC constraints. "
                f"Consider using approx=False to use power cones instead.",
                stacklevel=6,
            )

    return t, constrs


def _geo_mean_cone_canon(expr, args):
    x = args[0]
    w = expr.w
    shape = expr.shape
    t = Variable(shape)

    if x.shape == ():
        x_list = [x]
    else:
        x_list = [x[i] for i in range(len(w))]

    W = vstack(x_list)
    alpha = np.array([float(wi) for wi in w]).reshape(-1, 1)
    return t, [PowConeND(W, t, alpha, axis=0)]
