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

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
from typing import List, Tuple

import numpy as np


class ExpCone(Constraint):
    """A reformulated exponential cone constraint.

    Operates elementwise on :math:`x, y, z`.

    Original cone:

    .. math::

        K = \\{(x,y,z) \\mid y > 0, ye^{x/y} <= z\\}
            \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Reformulated cone:

    .. math::

        K = \\{(x,y,z) \\mid y, z > 0, y\\log(y) + x \\leq y\\log(z)\\}
             \\cup \\{(x,y,z) \\mid x \\leq 0, y = 0, z \\geq 0\\}

    Parameters
    ----------
    x : Variable
        x in the exponential cone.
    y : Variable
        y in the exponential cone.
    z : Variable
        z in the exponential cone.
    """

    def __init__(self, x, y, z, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        self.x = Expression.cast_to_const(x)
        self.y = Expression.cast_to_const(y)
        self.z = Expression.cast_to_const(z)
        super(ExpCone, self).__init__([self.x, self.y, self.z],
                                      constr_id)

    def __str__(self) -> str:
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    @property
    def residual(self):
        # TODO(akshayka): The projection should be implemented directly.
        from cvxpy import Problem, Minimize, Variable, norm2, hstack
        if self.x.value is None or self.y.value is None or self.z.value is None:
            return None
        x = Variable(self.x.shape)
        y = Variable(self.y.shape)
        z = Variable(self.z.shape)
        constr = [ExpCone(x, y, z)]
        obj = Minimize(norm2(hstack([x, y, z]) -
                             hstack([self.x.value, self.y.value, self.z.value])))
        problem = Problem(obj, constr)
        return problem.solve()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return 3 * self.num_cones()

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.x.size

    def cone_sizes(self) -> List[int]:
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """An exponential constraint is DCP if each argument is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
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
        # TODO(akshaya,SteveDiamond): verify that reshaping below works correctly
        value = np.reshape(value, newshape=(-1, 3))
        dv0 = np.reshape(value[:, 0], newshape=self.x.shape)
        dv1 = np.reshape(value[:, 1], newshape=self.y.shape)
        dv2 = np.reshape(value[:, 2], newshape=self.z.shape)
        self.dual_variables[0].save_value(dv0)
        self.dual_variables[1].save_value(dv1)
        self.dual_variables[2].save_value(dv2)
