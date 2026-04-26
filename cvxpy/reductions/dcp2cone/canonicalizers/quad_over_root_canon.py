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

import numpy as np

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.constraints.second_order import RSOC
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def quad_over_root_canon(expr, args, solver_context: SolverInfo | None = None):
    r"""Canonicalize quad_over_root to RSOC constraints.

    Represents :math:`t \geq (ax^2 + bx + c) / \sqrt{y}` via the epigraph
    constraints from the guide, using RSOC (2pq >= ||v||^2).
    """
    x = args[0]
    y = args[1]
    a, b, c, d = expr.a, expr.b, expr.c, expr.d

    t = Variable(nonneg=True)
    z = Variable(nonneg=True)

    # z^2 <= y  (i.e. z <= sqrt(y))
    # RSOC: 2*y*1 >= ||sqrt(2)*z||^2 = 2*z^2  =>  y >= z^2
    constraints = [
        RSOC(np.sqrt(2) * z, y, 1),
    ]
    if d is not None:
        constraints.append(x >= d)

    disc = b ** 2 - 4 * a * c

    if disc > 0:
        # Case 1: two real roots, factored form.
        # p(x) = a(x - r1)(x - r2) with r1 <= r2 <= d.
        r1 = (-b - np.sqrt(disc)) / (2 * a)
        r2 = (-b + np.sqrt(disc)) / (2 * a)

        s1 = x - r1  # >= 0 on x >= d
        s2 = x - r2  # >= 0 on x >= d
        w = Variable(nonneg=True)

        # s1 * s2 >= w^2
        # RSOC: 2*s1*s2 >= ||sqrt(2)*w||^2
        constraints.append(RSOC(np.sqrt(2) * w, s1, s2))

        # t * z >= a * w^2
        # RSOC: 2*t*z >= ||sqrt(2*a)*w||^2
        constraints.append(RSOC(np.sqrt(2 * a) * w, t, z))
    else:
        # Case 2: complete the square.
        # p(x) = a(x + b/(2a))^2 + delta, where delta = c - b^2/(4a) >= 0.
        delta = c - b ** 2 / (4 * a)
        u_expr = np.sqrt(a) * (x + b / (2 * a))

        if delta > 1e-15:
            # t * z >= u^2 + delta
            # RSOC: 2*t*z >= ||sqrt(2)*u, sqrt(2*delta)||^2 = 2*u^2 + 2*delta
            constraints.append(
                RSOC(hstack([np.sqrt(2) * u_expr, np.sqrt(2 * delta)]), t, z)
            )
        else:
            # t * z >= u^2
            # RSOC: 2*t*z >= ||sqrt(2)*u||^2
            constraints.append(RSOC(np.sqrt(2) * u_expr, t, z))

    return t, constraints
