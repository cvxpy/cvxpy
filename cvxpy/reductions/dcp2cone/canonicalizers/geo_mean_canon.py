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

import numpy as np

from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.power import PowConeND
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def geo_mean_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize GeoMean to PowConeND constraints.

    Always produces PowConeND constraints with allow_approx flag set based on
    the atom's allow_approx attribute. If the solver doesn't support power cones
    and allow_approx=True, ApproxCone2Cone will convert to SOC.
    """
    x = args[0]
    w = expr.w
    allow_approx = getattr(expr, 'allow_approx', False)

    # Single non-zero weight: geo_mean is just that element (affine).
    if len(w) == 1:
        # Ensure shape matches: geo_mean always returns a scalar, but x may be (1,)
        if expr.shape == () and x.shape == (1,):
            return x[0], []
        return x, []

    shape = expr.shape
    t = Variable(shape)

    if x.shape == ():
        x_list = [x]
    else:
        x_list = [x[i] for i in range(len(w))]

    W = vstack(x_list)
    alpha = np.array([float(wi) for wi in w]).reshape(-1, 1)
    return t, [PowConeND(W, t, alpha, axis=0, allow_approx=allow_approx)]
