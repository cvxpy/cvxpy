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

class ExpCone(NonlinearConstraint):
    """An exponential cone constraint.

    K = {(x,y,z) | y > 0, ye^(x/y) <= z} U {(x,y,z) | x <= 0, y = 0, z >= 0}

    Attributes:
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
        super(ExpCone, self).__init__(self._solver_hook,
                                      [self.x, self.y, self.z])

    def __str__(self):
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    def format(self):
        """Formats EXP constraints as inequalities for the solver.
        """
        return [self.x >= 0, self.y >= 0, self.z >= 0]

    @staticmethod
    def _solver_hook(vars_=None, scaling=None):
        """A function used by CVXOPT's nonlinear solver.

        Based on f(x,y,z) = ye^(x/y) - z.

        want y e^(x/y) - z <= 0
        equiv to
            x/y - log(z) + log(y) <= 0

        Args:
            vars_: A cvxopt dense matrix with values for (x,y,z).
            scaling: A scaling for the Hessian.

        Returns:
            _solver_hook() returns the constraint size and a feasible point.
            _solver_hook(x) returns the function value and gradient at x.
            _solver_hook(x, z) returns the function value, gradient,
            and (z scaled) Hessian at x.
        """
        if vars_ is None:
            return ExpCone.size[0], cvxopt.matrix([0.0, 1.0, 1.0])
        # Unpack vars_
        x, y, z = vars_
        # Out of domain.
        if y < 0.0 or (y == 0.0 and x > 0.0) or z < 0.0:
            return None
        # Evaluate the function.
        f = x/y - math.log(z) + math.log(y)
        # Compute the gradient.
        Df = cvxopt.matrix([1.0/y,
                            (y-x)/(y**2),
                            -1.0/z]).T
        if scaling is None:
            return f, Df
        # Compute the Hessian.
        H = cvxopt.matrix([
                [0, -1.0/(y**2), 0.0],
                [-1.0/(y**2), (2.0*x - y)/(y**3), 0.0],
                [0.0, 0.0, 1.0/(z**2)],
            ])
        return f, Df, scaling*H
