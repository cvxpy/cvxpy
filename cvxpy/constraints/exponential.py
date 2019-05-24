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

    def __init__(self, x, y, z, constr_id=None):
        self.x = x
        self.y = y
        self.z = z
        super(ExpCone, self).__init__([self.x, self.y, self.z],
                                      constr_id)

    def __str__(self):
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    def __repr__(self):
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
    def size(self):
        """The number of entries in the combined cones.
        """
        # TODO use size of dual variable(s) instead.
        return sum(self.cone_sizes())

    def num_cones(self):
        """The number of elementwise cones.
        """
        return np.prod(self.args[0].shape, dtype=int)

    def cone_sizes(self):
        """The dimensions of the exponential cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [3]*self.num_cones()

    def is_dcp(self):
        """An exponential constraint is DCP if each argument is affine.
        """
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self):
        return False

    def is_dqcp(self):
        return self.is_dcp()

    def canonicalize(self):
        """Canonicalizes by converting expressions to LinOps.
        """
        arg_objs = []
        arg_constr = []
        for arg in self.args:
            arg_objs.append(arg.canonical_form[0])
            arg_constr + arg.canonical_form[1]
        return 0, [ExpCone(*arg_objs)] + arg_constr
