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


def _initial_point(y, x):
    """Rescale entries with radius below MIN_INIT_RADIUS to that radius,
    preserving their angle. Entries at the origin are moved to angle 0."""
    r = np.hypot(x, y)
    at_origin = (r == 0)
    scale = np.where(r < MIN_INIT_RADIUS, MIN_INIT_RADIUS / np.where(at_origin, 1.0, r), 1.0)
    y = y * scale
    x = np.where(at_origin, MIN_INIT_RADIUS, x * scale)
    return y, x


def atan2_canon(expr, args):
    """Canonicalize atan2(y, x) by lifting both arguments to fresh variables
    (required by the diff engine), initialized away from the origin."""
    shape = expr.shape
    y0 = np.zeros(shape) if args[0].value is None else np.broadcast_to(args[0].value, shape)
    x0 = np.zeros(shape) if args[1].value is None else np.broadcast_to(args[1].value, shape)
    y0, x0 = _initial_point(y0, x0)
    t1 = Variable(shape)
    t2 = Variable(shape)
    t1.value = y0
    t2.value = x0
    return expr.copy([t1, t2]), [t1 == args[0], t2 == args[1]]
