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

import cvxpy.settings as s
from cvxpy.error import SolverError
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.lin_ops.lin_op import VARIABLE
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.nonlinear import NonlinearConstraint
from cvxpy.constraints.utilities import format_elemwise
import numpy as np
import math


class ExpCone(NonlinearConstraint):
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
        super(ExpCone, self).__init__(self._solver_hook,
                                      [self.x, self.y, self.z],
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

    def format(self, eq_constr, leq_constr, dims, solver):
        """Formats EXP constraints for the solver.

        Parameters
        ----------
        eq_constr : list
            A list of the equality constraints in the canonical problem.
        leq_constr : list
            A list of the inequality constraints in the canonical problem.
        dims : dict
            A dict with the dimensions of the conic constraints.
        solver : str
            The solver being called.
        """
        if solver.name() == s.CVXOPT:
            eq_constr += self.__CVXOPT_format[0]
        elif solver.name() in [s.SCS, s.JULIA_OPT]:
            leq_constr += self.__SCS_format[1]
        elif solver.name() == s.ECOS:
            leq_constr += self.__ECOS_format[1]
        else:
            raise SolverError("Solver does not support exponential cone.")
        # Update dims.
        dims[s.EXP_DIM] += self.num_cones()

    @property
    def size(self):
        """The number of entries in the combined cones.
        """
        # TODO use size of dual variable(s) instead.
        return sum(self.cone_sizes())

    # @property
    # def shape(self):
    #     """Represented as vectorized.
    #     """
    #     return (self.size,)

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

    def canonicalize(self):
        """Canonicalizes by converting expressions to LinOps.
        """
        arg_objs = []
        arg_constr = []
        for arg in self.args:
            arg_objs.append(arg.canonical_form[0])
            arg_constr + arg.canonical_form[1]
        return 0, [ExpCone(*arg_objs)] + arg_constr

    @pu.lazyprop
    def __ECOS_format(self):
        return ([], format_elemwise([self.x, self.z, self.y]))

    @pu.lazyprop
    def __SCS_format(self):
        return ([], format_elemwise([self.x, self.y, self.z]))

    @pu.lazyprop
    def __CVXOPT_format(self):
        constraints = []
        for i, var in enumerate(self.vars_):
            if var.type is not VARIABLE:
                lone_var = lu.create_var(var.shape)
                constraints.append(lu.create_eq(lone_var, var))
                self.vars_[i] = lone_var
        return (constraints, [])

    def _solver_hook(self, vars_=None, scaling=None):
        """A function used by CVXOPT's nonlinear solver.

        Based on f(x,y,z) = y * log(y) + x - y * log(z).

        Parameters
        ----------
            vars_: A cvxopt dense matrix with values for (x,y,z).
            scaling: A scaling for the Hessian.

        Returns
        -------
            _solver_hook() returns the constraint shape and a feasible point.
            _solver_hook(x) returns the function value and gradient at x.
            _solver_hook(x, z) returns the function value, gradient,
            and (z scaled) Hessian at x.
        """
        import cvxopt  # Not necessary unless using cvxopt solver.
        entries = int(np.prod(self.shape))
        if vars_ is None:
            x_init = entries*[0.0]
            y_init = entries*[0.5]
            z_init = entries*[1.0]
            return entries, cvxopt.matrix(x_init + y_init + z_init)
        # Unpack vars_
        x = vars_[0:entries]
        y = vars_[entries:2*entries]
        z = vars_[2*entries:]
        # Out of domain.
        # TODO what if y == 0.0?
        if min(y) <= 0.0 or min(z) <= 0.0:
            return None
        # Evaluate the function.
        f = cvxopt.matrix(0., (entries, 1))
        for i in range(entries):
            f[i] = x[i] - y[i]*math.log(z[i]) + y[i]*math.log(y[i])
        # Compute the gradient.
        Df = cvxopt.matrix(0., (entries, 3*entries))
        for i in range(entries):
            Df[i, i] = 1.0
            Df[i, entries+i] = math.log(y[i]) - math.log(z[i]) + 1.0
            Df[i, 2*entries+i] = -y[i]/z[i]

        if scaling is None:
            return f, Df
        # Compute the Hessian.
        big_H = cvxopt.spmatrix(0, [], [], size=(3*entries, 3*entries))
        for i in range(entries):
            H = cvxopt.matrix([
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0/y[i], -1.0/z[i]],
                    [0.0, -1.0/z[i], y[i]/(z[i]**2)],
                ])
            big_H[i:3*entries:entries, i:3*entries:entries] = scaling[i]*H
        return f, Df, big_H
