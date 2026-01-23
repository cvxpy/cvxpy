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

from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.power import PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
from cvxpy.utilities.solver_context import SolverInfo


def geo_mean_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    w = expr.w
    shape = expr.shape
    t = Variable(shape)

    if x.shape == ():
        x_list = [x]
    else:
        x_list = [x[i] for i in range(len(w))]

    # Check if user requested power cones (approx=False)
    if not expr._approx:
        solver_supports_powcone = (
            solver_context is not None
            and PowConeND in solver_context.solver_supported_constraints
        )
        if solver_supports_powcone:
            # Use PowConeND: prod(W^alpha) >= |z|
            # Stack x_list into a column vector W
            W = vstack(x_list)
            alpha = np.array([float(wi) for wi in w]).reshape(-1, 1)
            return t, [PowConeND(W, t, alpha, axis=0)]

    # Use SOC approximation
    constrs = gm_constrs(t, x_list, w)

    # Warn if the solver supports power cones and the approximation is poor
    solver_supports_powcone = (
        solver_context is not None
        and PowConeND in solver_context.solver_supported_constraints
    )
    if solver_supports_powcone and expr._approx:
        approx_error = getattr(expr, 'approx_error', 0.0)
        num_soc = len(constrs)
        if approx_error > 1e-6 or num_soc > 4:
            warnings.warn(
                f"geo_mean is being approximated (error: {approx_error:.2e}) "
                f"using {num_soc} SOC constraints. "
                f"Consider using approx=False to use power cones instead.",
                stacklevel=6
            )

    return t, constrs
