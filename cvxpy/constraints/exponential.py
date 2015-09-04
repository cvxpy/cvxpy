"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.error import SolverError
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.lin_ops.lin_op import VARIABLE
import cvxpy.utilities.performance_utils as pu
from cvxpy.constraints.nonlinear import NonlinearConstraint
from cvxpy.constraints.utilities import format_elemwise
import math
import cvxopt

class ExpCone(NonlinearConstraint):
    """A reformulated exponential cone constraint.

    Operates elementwise on x, y, z.

    Original cone:
    K = {(x,y,z) | y > 0, ye^(x/y) <= z}
         U {(x,y,z) | x <= 0, y = 0, z >= 0}
    Reformulated cone:
    K = {(x,y,z) | y, z > 0, y * log(y) + x <= y * log(z)}
         U {(x,y,z) | x <= 0, y = 0, z >= 0}

    Attributes
    ----------
        x: Variable x in the exponential cone.
        y: Variable y in the exponential cone.
        z: Variable z in the exponential cone.
    """
    CVXOPT_DENSE_INTF = intf.get_matrix_interface(cvxopt.matrix)
    CVXOPT_SPARSE_INTF = intf.get_matrix_interface(cvxopt.spmatrix)

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.size = self.x.size
        super(ExpCone, self).__init__(self._solver_hook,
                                      [self.x, self.y, self.z])

    def __str__(self):
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

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
        elif solver.name() == s.SCS:
            leq_constr += self.__SCS_format[1]
        elif solver.name() == s.ECOS:
            leq_constr += self.__ECOS_format[1]
        else:
            raise SolverError("Solver does not support exponential cone.")
        # Update dims.
        dims[s.EXP_DIM] += self.size[0]*self.size[1]

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
            if not var.type is VARIABLE:
                lone_var = lu.create_var(var.size)
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
            _solver_hook() returns the constraint size and a feasible point.
            _solver_hook(x) returns the function value and gradient at x.
            _solver_hook(x, z) returns the function value, gradient,
            and (z scaled) Hessian at x.
        """
        entries = self.size[0]*self.size[1]
        if vars_ is None:
            x_init = entries*[0.0]
            y_init = entries*[0.5]
            z_init = entries*[1.0]
            return self.size[0], cvxopt.matrix(x_init + y_init + z_init)
        # Unpack vars_
        x = vars_[0:entries]
        y = vars_[entries:2*entries]
        z = vars_[2*entries:]
        # Out of domain.
        # TODO what if y == 0.0?
        if min(y) <= 0.0 or min(z) <= 0.0:
            return None
        # Evaluate the function.
        f = self.CVXOPT_DENSE_INTF.zeros(entries, 1)
        for i in range(entries):
            f[i] = x[i] - y[i]*math.log(z[i]) + y[i]*math.log(y[i])
        # Compute the gradient.
        Df = self.CVXOPT_DENSE_INTF.zeros(entries, 3*entries)
        for i in range(entries):
            Df[i, i] = 1.0
            Df[i, entries+i] = math.log(y[i]) - math.log(z[i]) + 1.0
            Df[i, 2*entries+i] = -y[i]/z[i]

        if scaling is None:
            return f, Df
        # Compute the Hessian.
        big_H = self.CVXOPT_SPARSE_INTF.zeros(3*entries, 3*entries)
        for i in range(entries):
            H = cvxopt.matrix([
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0/y[i], -1.0/z[i]],
                    [0.0, -1.0/z[i], y[i]/(z[i]**2)],
                ])
            big_H[i:3*entries:entries, i:3*entries:entries] = scaling[i]*H
        return f, Df, big_H
