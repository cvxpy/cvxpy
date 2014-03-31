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

from nonlinear import NonlinearConstraint
import math
import cvxopt

def get_solver_hook(func_oracle):
    """Returns a customized solver_hook.

    Parameters
    ----------
        func_oracle: A function to compute the value, gradient, and Hessian.

    Returns
    -------
        A function for the solver to call.
    """
    def solver_hook(vars_=None, scaling=None):
        """A function used by CVXOPT's nonlinear solver.

        Based on f(x,y,z) = ye^(x/y) - z.

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
        if vars_ is None:
            return ExpCone.size[0], cvxopt.matrix([0.0, 0.5, 1.0])
        # Unpack vars_
        x, y, z = vars_
        # Out of domain.
        if y < 0.0 or y == 0.0 and (x > 0.0 or z < 0.0):
            return None
        f, Df, H = func_oracle(x, y, z)
        if scaling is None:
            return f, Df
        else:
            return f, Df, scaling*H
    return solver_hook

class ExpCone(NonlinearConstraint):
    """An exponential cone constraint.

    K = {(x,y,z) | y > 0, ye^(x/y) <= z} U {(x,y,z) | x <= 0, y = 0, z >= 0}

    Attributes
    ----------
        x: The scalar variable x in the exponential cone.
        y: The scalar variable y in the exponential cone.
        z: The scalar variable z in the exponential cone.
        m: The dimension of the constraint.
        _constraints: A list of equality constraints connecting
                      new variables to the given x,y,z.
    """

    # The dimensions of the exponential cone.
    size = (1, 1)

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        solver_hook = get_solver_hook(self._func_oracle)
        super(ExpCone, self).__init__(solver_hook,
                                      [self.x, self.y, self.z])

    def __str__(self):
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    @staticmethod
    def _func_oracle(x, y, z):
        """Computes the function value, gradient, and Hessian.

        Parameters
        ----------
            x: float
                The given value of x.
            y: float
                The given value of y.
            z: float
                The given value of z.

        Returns
        -------
        tuple
            (function value, gradient cvxopt matrix, Hessian cvxopt matrix)
        """
        # Evaluate the function.
        f = y*math.exp(x/y) - z
        # Compute the gradient.
        Df = cvxopt.matrix([math.exp(x/y),
                            math.exp(x/y)*(1-x/y),
                            -1]).T
        # Compute the Hessian.
        H = math.exp(x/y)*cvxopt.matrix([
                [1.0/y, -x/y**2, 0.0],
                [-x/y**2, x**2/y**3, 0.0],
                [0.0, 0.0, 0.0],
            ])
        return (f, Df, H)

class LogCone(ExpCone):
    """A reformulated exponential cone constraint.

    K = {(x,y,z) | y > 0, y * log(y) + x <= y * log(z)}
         U {(x,y,z) | x <= 0, y = 0, z >= 0}
    """

    @staticmethod
    def _func_oracle(x, y, z):
        """Computes the function value, gradient, and Hessian.

        Parameters
        ----------
            x: float
                The given value of x.
            y: float
                The given value of y.
            z: float
                The given value of z.

        Returns
        -------
        tuple
            (function value, gradient cvxopt matrix, Hessian cvxopt matrix)
        """
        # Evaluate the function.
        f = x - y*math.log(z) + y*math.log(y)
        # Compute the gradient.
        Df = cvxopt.matrix([1.0,
                            math.log(y) - math.log(z) + 1.0,
                            -y/z]).T
        # Compute the Hessian.
        H = cvxopt.matrix([
                [0.0, 0.0, 0.0],
                [0.0, 1.0/y, -1.0/z],
                [0.0, -1.0/z, y/(z**2)],
            ])
        return (f, Df, H)
