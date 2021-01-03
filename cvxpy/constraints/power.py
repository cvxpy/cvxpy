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
        self.x = Expression.cast_to_const(x) #TODO: add checks that inputs are real
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        self.alpha = Expression.cast_to_const(alpha)  # cheat!!
        super(PowerCone3D, self).__init__([self.x, self.y, self.z, self.alpha],
                                      constr_id)
        pass

    # Override base class
    def _construct_dual_variables(self, args):
        self.dual_variables = [cvxtypes.variable()(arg.shape) for arg in args[:3]]

    # Override base class
    def is_imag(self):
        return False

    # Override base class
    def is_complex(self):
        return False

    @property
    def size(self):
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def cone_sizes(self):
        """The dimensions of the power cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self, dpp=False):
        """A power cone constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args[:3])
        return all(arg.is_affine() for arg in self.args[:3])

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
        # TODO(rileyjmurray): figure out why the reshaping had to be done differently,
        #   relative to ExpCone constraints.



class PowerConeND(Constraint):

    def __init__(self, w, z, alpha):
        """
        \\prod_i w_i^{\\alpha_i} >= |z|
        w >= 0
        """
        pass
