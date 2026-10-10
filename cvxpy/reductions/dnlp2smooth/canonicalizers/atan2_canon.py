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


def _value_on_shape(arg, shape):
    """Current value of arg broadcast to shape, or zeros if it has none."""
    if arg.value is None:
        return np.zeros(shape)
    return np.array(np.broadcast_to(arg.value, shape), dtype=float)


def _initial_point(y, x):
    """Push entries closer to the origin than MIN_INIT_RADIUS out to that
    radius, keeping their angle. The origin itself goes to angle 0."""
    r = np.hypot(x, y)
    at_origin = r == 0
    scale = np.where(r < MIN_INIT_RADIUS, MIN_INIT_RADIUS / np.where(at_origin, 1.0, r), 1.0)
    y = y * scale
    x = np.where(at_origin, MIN_INIT_RADIUS, x * scale)
    return y, x


def atan2_canon(expr, args):
    """Canonicalize atan2(y, x) by lifting both arguments to fresh variables.

    The diff engine's atan2 atom is leaf-only: both children must be distinct
    variables of the same shape. Lifting through equality constraints handles
    non-variable arguments, scalar/matrix broadcasting and atan2(x, x).

    The derivatives of atan2 are undefined at the origin, and variables without
    a user-specified value are initialized to zero before this reduction runs,
    so the fresh variables are initialized away from the origin.
    """
    shape = expr.shape
    y0, x0 = _initial_point(_value_on_shape(args[0], shape),
                            _value_on_shape(args[1], shape))
    t1 = Variable(shape)
    t2 = Variable(shape)
    t1.value = y0
    t2.value = x0
    return expr.copy([t1, t2]), [t1 == args[0], t2 == args[1]]
