"""
Copyright 2013 Steven Diamond, Eric Chu

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

from .. atom import Atom
from ..elementwise.elementwise import Elementwise
from cvxpy.expressions.variables import Variable
from cvxpy.constraints.nonlinear import NonlinearConstraint
import cvxpy.utilities as u
import cvxpy.interface as intf

import numpy as np
import cvxopt

# TODO: negative log func, doesn't work with matrix variables (yet)
def neg_log_func(m):
    # m is the size of t1 (and t2)
    # input is 2*m, output is m
    def F(x=None, z=None):
        # x = (t1, t2)
        # t1 - log(t2) <= 0
        if x is None: return m, cvxopt.matrix(m*[0.0] + m*[1.0])
        if min(x[m:]) <= 0.0: return None
        f = x[0:m] - cvxopt.log(x[m:])
        Df = cvxopt.sparse([[cvxopt.spdiag(cvxopt.matrix(1.0, (m,1)))], [cvxopt.spdiag(-(x[m:]**-1))]])
        if z is None: return f, Df
        ret = cvxopt.mul(z, x[m:]**-2)
        # TODO: add regularization for the Hessian?
        H = cvxopt.spdiag(cvxopt.matrix([cvxopt.matrix(0, (m,1)), ret]))
        return f, Df, H
    return F

class log(Elementwise):
    """ Elementwise logarithm. """
    def __init__(self, x):
        super(log, self).__init__(x)

    # Returns the elementwise natural log of x.
    @Atom.numpy_numeric
    def numeric(self, values):
        return np.log(values[0])

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise Exception("The argument '%s' to log must resolve to a vector."
                % self.args[0].name())

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.INCREASING]

    def graph_implementation(self, arg_objs):
        """ any expression that involves log

                f*log(a*x + b) + g

            becomes

                f*t1 + g
                t1 - log(t2) <= 0  # this is always homogeneous
                t2 = a*x + b

            even if the argument is just a single variable
        """
        x = arg_objs[0]
        t1 = Variable(*self.size)
        t2 = Variable(*self.size)
        constraints = [
            NonlinearConstraint(neg_log_func(self.size[0]*self.size[1]), 
                                [t1,t2]),
            x == t2,
        ]

        return (t1, constraints)