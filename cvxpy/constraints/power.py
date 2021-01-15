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
import numpy as np


class PowerCone3D(Constraint):

    def __init__(self, x, y, z, alpha, constr_id=None):
        """
        x[i]**alpha * y[i]**(1-alpha) >= |z[i]|  for all i
        x >= 0, y >= 0
        """
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)  # TODO: add checks that inputs are real
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        self.alpha = Expression.cast_to_const(alpha)  # cheat!!
        super(PowerCone3D, self).__init__([self.x, self.y, self.z],
                                          constr_id)
        pass

    def __str__(self):
        return "Pow3D(%s, %s, %s; %s)" % (self.x, self.y, self.z, self.alpha)

    def residual(self):
        # TODO: implement
        raise NotImplementedError()

    def get_data(self):
        return [self.alpha]

    def is_imag(self):
        return False

    def is_complex(self):
        return False

    @property
    def size(self):
        return 3 * self.num_cones()

    def num_cones(self):
        return self.x.size

    def cone_sizes(self):
        return [3]*self.num_cones()

    def is_dcp(self, dpp=False):
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp=False):
        return False

    def is_dqcp(self):
        return self.is_dcp()

    @property
    def shape(self):
        s = (3,) + self.x.shape
        return s

    def save_dual_value(self, value):
        value = np.reshape(value, newshape=(3, -1))
        dv0 = np.reshape(value[0, :], newshape=self.x.shape)
        dv1 = np.reshape(value[1, :], newshape=self.y.shape)
        dv2 = np.reshape(value[2, :], newshape=self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)
        # TODO: figure out why the reshaping had to be done differently,
        #   relative to ExpCone constraints.


class PowerConeND(Constraint):

    _TOL_ = 1e-6

    def __init__(self, W, z, alpha, axis=0, constr_id=None):
        """
        \\prod_i w_i^{\\alpha_i} >= |z|
        w >= 0
        """
        Expression = cvxtypes.expression()
        W = Expression.cast_to_const(W)
        if not (W.is_real() and W.is_affine()):
            raise ValueError("Invalid first argument.")
        z = Expression.cast_to_const(z)
        if z.ndim > 1 or not (z.is_real() and z.is_affine()):
            raise ValueError("Invalid second argument.")
        # Check t has one entry per cone.
        if (W.ndim <= 1 and z.size > 1) or \
           (W.ndim == 2 and z.size != W.shape[1-axis]) or \
           (W.ndim == 1 and axis == 1):
            raise ValueError(
                "Argument dimensions %s and %s, with axis=%i, are incompatible."
                % (W.shape, z.shape, axis))
        if W.ndim == 2 and W.shape[axis] <= 1:
            msg = "PowerConeND requires left-hand-side to have at least two terms."
            raise ValueError(msg)
        alpha = Expression.cast_to_const(alpha)
        if alpha.shape != W.shape:
            raise ValueError("Argument dimensions %s and %s are not equal."
                             % (W.shape, alpha.shape))
        if np.any(alpha.value <= 0):
            raise ValueError("Argument alpha must be entry-wise positive.")
        if np.any(np.abs(1 - np.sum(alpha.value, axis=axis)) > PowerConeND._TOL_):
            raise ValueError("Argument alpha must sum to 1 along axis %s." % axis)
        self.W = W
        self.z = z
        self.alpha = alpha
        self.axis = axis
        if z.ndim == 0:
            z = z.flatten()
        super(PowerConeND, self).__init__([W, z], constr_id)

    def __str__(self):
        return "PowND(%s, %s; %s)" % (self.W, self.z, self.alpha)

    def is_imag(self):
        return False

    def is_complex(self):
        return False

    def get_data(self):
        return [self.alpha, self.axis]

    @property
    def residual(self):
        # TODO: implement
        raise NotImplementedError()

    def num_cones(self):
        return self.z.size

    @property
    def size(self):
        cone_size = 1 + self.args[0].shape[self.axis]
        return cone_size * self.num_cones()

    def cone_sizes(self):
        cone_size = 1 + self.args[0].shape[self.axis]
        return [cone_size] * self.num_cones()

    def is_dcp(self, dpp=False):
        """A power cone constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                args_ok = self.args[0].is_affine() and self.args[1].is_affine()
                exps_ok = not isinstance(self.alpha, cvxtypes.parameter())
                return args_ok and exps_ok
        return True

    def is_dgp(self, dpp=False):
        return False

    def is_dqcp(self):
        return self.is_dcp()

    def save_dual_value(self, value):
        # TODO: implement
        pass
