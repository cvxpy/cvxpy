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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.solver_context import SolverInfo


def pnorm_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize Pnorm to power cone or SOC constraints.

    For p=2, uses SOC directly (exact).
    For other p, produces PowCone3D constraints with allow_approx flag set based on
    the atom's allow_approx attribute. If the solver doesn't support power cones
    and allow_approx=True, ApproxCone2Cone will convert to SOC.
    """
    p = expr.p
    x = args[0]
    axis = expr.axis
    shape = expr.shape
    allow_approx = getattr(expr, 'allow_approx', False)

    # p == 2 case: use SOC directly (exact, no approximation needed)
    if p == 2:
        t = Variable(shape)
        if axis is None:
            assert shape == tuple()
            return t, [SOC(t, vec(x, order="F"))]
        else:
            return t, [SOC(vec(t, order="F"), x, axis)]

    # For other p values, use power cone constraints
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
            PowCone3D(vec(x, order="F"), vec(r, order="F"), vec(promoted_t, order="F"), alpha,
                      allow_approx=allow_approx)
        ]
    elif 0 < p < 1:
        alpha = float(p)
        constraints += [
            PowCone3D(vec(x, order="F"), vec(promoted_t, order="F"), vec(r, order="F"), alpha,
                      allow_approx=allow_approx)
        ]
    elif p > 1:
        alpha = float(1 / p)
        constraints += [
            PowCone3D(vec(r, order="F"), vec(promoted_t, order="F"), vec(x, order="F"), alpha,
                      allow_approx=allow_approx)
        ]

    return t, constraints
