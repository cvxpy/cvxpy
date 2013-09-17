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
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint
from cvxpy.constraints.nonlinear import NonlinearConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

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

class log(Atom):
    """ Elementwise logarithm. """
    
    def __init__(self, x):
        super(log, self).__init__(x)
                
    # The shape is the common shape of all the arguments.
    def set_shape(self):
        self.validate_arguments()
        self._shape = self.args[0].shape
    
    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise Exception("The argument '%s' to norm1 must resolve to a vector." 
                % self.args[0].name())

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.Monotonicity.INCREASING]
        
    @staticmethod
    def graph_implementation(var_args, size):
        """ any expression that involves log
        
                f*log(a*x + b) + g
                
            becomes
            
                f*t1 + g
                t1 - log(t2) <= 0  # this is always homogeneous
                t2 = a*x + b
            
            even if the argument is just a single variable
        """
        x = var_args[0]
        t1 = Variable(*size)      
        t2 = Variable(*size)
        constraints = [
            NonlinearConstraint(neg_log_func(size[0]*size[1]),[t1,t2]),
            AffEqConstraint(x,t2)
        ]
        
        return (t1, constraints)

    # Return the log of the arguments' elements at the given index.
    def index_object(self, key):
        args = []
        for arg in self.args:
            if arg.size == (1,1):
                args.append(arg)
            else:
                args.append(arg[key])
        return self.__class__(*args)