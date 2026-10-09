"""
Copyright 2025 CVXPY developers

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

from cvxpy.expressions.variable import Variable

# Initial points closer to the origin than this radius are pushed out to it.
# The derivatives of atan2 blow up like 1/r, so a unit radius keeps the
# gradient and Hessian O(1) at the starting point (cf. quad_over_lin_canon).
MIN_INIT_RADIUS = 1.0


def _lifted_value(arg, shape):
    """Value of arg broadcast to shape, or zeros if it has no value."""
    if arg.value is None:
        return np.zeros(shape)
    return np.array(np.broadcast_to(arg.value, shape), dtype=float)


def atan2_canon(expr, args):
    """Canonicalize atan2(y, x) by lifting both arguments to fresh variables.

    The diff engine's atan2 atom is leaf-only: both children must be distinct
    variables of the same shape. Lifting through equality constraints handles
    non-variable arguments, scalar/matrix broadcasting and atan2(x, x).

    The derivatives of atan2 are undefined at the origin, and variables without
    a user-specified value are initialized to zero before this reduction runs.
    The fresh variables are therefore initialized at the argument values pushed
    out to radius MIN_INIT_RADIUS (preserving the angle; the exact origin is
    moved to angle 0) without touching the user's variables.
    """
    shape = expr.shape
    y0 = _lifted_value(args[0], shape)
    x0 = _lifted_value(args[1], shape)

    r = np.hypot(x0, y0)
    small = r < MIN_INIT_RADIUS
    at_origin = r == 0
    scale = MIN_INIT_RADIUS / np.where(at_origin, 1.0, r)
    x0 = np.where(small, np.where(at_origin, MIN_INIT_RADIUS, x0 * scale), x0)
    y0 = np.where(small, y0 * scale, y0)

    t1 = Variable(shape)
    t2 = Variable(shape)
    t1.value = y0
    t2.value = x0
    return expr.copy([t1, t2]), [t1 == args[0], t2 == args[1]]
