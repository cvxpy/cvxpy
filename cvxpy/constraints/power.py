"""
Copyright 2021 the CVXPY developers

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

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
from typing import List, Tuple

import numpy as np


class PowCone3D(Constraint):
    """
    An object representing a collection of 3D power cone constraints

        x[i]**alpha[i] * y[i]**(1-alpha[i]) >= |z[i]|  for all i
        x >= 0, y >= 0

    If the parameter alpha is a scalar, it will be promoted to
    a vector matching the (common) sizes of x, y, z. The numeric
    value of alpha (or its components, in the vector case) must
    be a number in the open interval (0, 1).

    We store flattened representations of the arguments (x, y, z,
    and alpha) as Expression objects. We construct dual variables
    with respect to these flattened representations.
    """

    def __init__(self, x, y, z, alpha, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        for val in [self.x, self.y, self.z]:
            if not (val.is_affine() and val.is_real()):
                raise ValueError('All arguments must be affine and real.')
        alpha = Expression.cast_to_const(alpha)
        if alpha.is_scalar():
            alpha = cvxtypes.promote()(alpha, self.x.shape)
        self.alpha = alpha
        if np.any(self.alpha.value <= 0) or np.any(self.alpha.value >= 1):
            msg = "Argument alpha must have entries in the open interval (0, 1)."
            raise ValueError(msg)
        arg_sizes = [self.x.size, self.y.size, self.z.size, self.alpha.size]
        if min(arg_sizes) != max(arg_sizes):
            msg = ("All arguments must have the same size. Provided arguments are"
                   "of size %s" % str(arg_sizes))
            raise ValueError(msg)
        super(PowCone3D, self).__init__([self.x, self.y, self.z],
                                        constr_id)

    def __str__(self) -> str:
        return "Pow3D(%s, %s, %s; %s)" % (self.x, self.y, self.z, self.alpha)

    def residual(self):
        # TODO: The projection should be implemented directly.
        from cvxpy import Problem, Minimize, Variable, norm2, hstack
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [PowCone3D(x, y, z, self.alpha)]
        obj = Minimize(norm2(hstack([x, y, z]) -
                             hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve(solver='SCS', eps=1e-8)

    def get_data(self):
        return [self.alpha]

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return 3 * self.num_cones()

    def num_cones(self):
        return self.x.size

    def cone_sizes(self) -> List[int]:
        return [3]*self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                args_ok = all(arg.is_affine() for arg in self.args)
                exps_ok = not isinstance(self.alpha, cvxtypes.parameter())
                return args_ok and exps_ok
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def shape(self) -> Tuple[int, ...]:
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value) -> None:
        value = np.reshape(value, newshape=(3, -1))
        dv0 = np.reshape(value[0, :], newshape=self.x.shape)
        dv1 = np.reshape(value[1, :], newshape=self.y.shape)
        dv2 = np.reshape(value[2, :], newshape=self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)
        # TODO: figure out why the reshaping had to be done differently,
        #   relative to ExpCone constraints.


class PowConeND(Constraint):
    """
    Represents a collection of N-dimensional power cone constraints
    that is *mathematically* equivalent to the following code
    snippet (which makes incorrect use of numpy functions on cvxpy
    objects):

        np.prod(np.power(W, alpha), axis=axis) >= np.abs(z),
        W >= 0

    All arguments must be Expression-like, and z must satisfy
    z.ndim <= 1. The columns (rows) of alpha must sum to 1 when
    axis=0 (axis=1).

    Note: unlike PowCone3D, we make no attempt to promote
    alpha to the appropriate shape. The dimensions of W and
    alpha must match exactly.

    Note: Dual variables are not currently implemented for this type
    of constraint.
    """

    _TOL_ = 1e-6

    def __init__(self, W, z, alpha, axis: int = 0, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        W = Expression.cast_to_const(W)
        if not (W.is_real() and W.is_affine()):
            msg = "Invalid first argument; W must be affine and real."
            raise ValueError(msg)
        z = Expression.cast_to_const(z)
        if z.ndim > 1 or not (z.is_real() and z.is_affine()):
            msg = ("Invalid second argument. z must be affine, real, "
                   "and have at most one z.ndim <= 1.")
            raise ValueError(msg)
        # Check z has one entry per cone.
        if (W.ndim <= 1 and z.size > 1) or \
           (W.ndim == 2 and z.size != W.shape[1-axis]) or \
           (W.ndim == 1 and axis == 1):
            raise ValueError(
                "Argument dimensions %s and %s, with axis=%i, are incompatible."
                % (W.shape, z.shape, axis))
        if W.ndim == 2 and W.shape[axis] <= 1:
            msg = "PowConeND requires left-hand-side to have at least two terms."
            raise ValueError(msg)
        alpha = Expression.cast_to_const(alpha)
        if alpha.shape != W.shape:
            raise ValueError("Argument dimensions %s and %s are not equal."
                             % (W.shape, alpha.shape))
        if np.any(alpha.value <= 0):
            raise ValueError("Argument alpha must be entry-wise positive.")
        if np.any(np.abs(1 - np.sum(alpha.value, axis=axis)) > PowConeND._TOL_):
            raise ValueError("Argument alpha must sum to 1 along axis %s." % axis)
        self.W = W
        self.z = z
        self.alpha = alpha
        self.axis = axis
        if z.ndim == 0:
            z = z.flatten()
        super(PowConeND, self).__init__([W, z], constr_id)

    def __str__(self) -> str:
        return "PowND(%s, %s; %s)" % (self.W, self.z, self.alpha)

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def get_data(self):
        return [self.alpha, self.axis]

    @property
    def residual(self):
        # TODO: The projection should be implemented directly.
        from cvxpy import Problem, Minimize, Variable, norm2, hstack
        if self.W.value is None or self.z.value is None:
            return None
        W = Variable(self.W.shape)
        z = Variable(self.z.shape)
        constr = [PowConeND(W, z, self.alpha, axis=self.axis)]
        obj = Minimize(norm2(hstack([W.flatten(), z.flatten()]) -
                             hstack([self.W.flatten().value, self.z.flatten().value])))
        problem = Problem(obj, constr)
        return problem.solve(solver='SCS', eps=1e-8)

    def num_cones(self):
        return self.z.size

    @property
    def size(self) -> int:
        cone_size = 1 + self.args[0].shape[self.axis]
        return cone_size * self.num_cones()

    def cone_sizes(self) -> List[int]:
        cone_size = 1 + self.args[0].shape[self.axis]
        return [cone_size] * self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """A power cone constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                args_ok = self.args[0].is_affine() and self.args[1].is_affine()
                exps_ok = not isinstance(self.alpha, cvxtypes.parameter())
                return args_ok and exps_ok
        return True

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def save_dual_value(self, value) -> None:
        # TODO: implement
        pass
