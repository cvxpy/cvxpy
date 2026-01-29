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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import powcone_constrs
from cvxpy.utilities.solver_context import SolverInfo


def power_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize Power to power cone constraints.

    Always produces PowCone3D constraints with allow_approx flag set based on
    the atom's allow_approx attribute. If the solver doesn't support power cones
    and allow_approx=True, ApproxCone2Cone will convert to SOC.
    """
    x = args[0]
    p = expr.p_used
    allow_approx = getattr(expr, 'allow_approx', False)

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []

    t = Variable(shape)

    if 0 < p < 1:
        alpha = float(p)
        return t, powcone_constrs(t, [x, ones], alpha, allow_approx=allow_approx)
    elif p > 1:
        alpha = float(1 / p)
        constrs = powcone_constrs(x, [t, ones], alpha, allow_approx=allow_approx)
        if p % 2 != 0:
            constrs += [x >= 0]
        return t, constrs
    elif p < 0:
        alpha = float(p / (p - 1))
        return t, powcone_constrs(ones, [x, t], alpha, allow_approx=allow_approx)
    else:
        raise NotImplementedError("This power is not yet supported.")
