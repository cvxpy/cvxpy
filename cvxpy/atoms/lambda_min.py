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

from atom import Atom
from affine.transpose import transpose
from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..constraints.semi_definite import SDP
import numpy as np
from numpy import linalg as LA

class lambda_min(Atom):
    """ Miximum eigenvalue; :math:`\lambda_{\min}(A)`.
    
    """
    def __init__(self, A):
        super(lambda_min, self).__init__(A)

    # Returns the smallest eigenvalue of A.
    # Requires that A be symmetric.
    @Atom.numpy_numeric
    def numeric(self, values):
        if not (values[0].T == values[0]).all():
            raise Exception("lambda_min called on a non-symmetric matrix.")
        w,v = LA.eig(values[0])
        return min(w)

    # Resolves to a scalar.
    def shape_from_args(self):
        return u.Shape(1,1)

    # Verify that the argument A is square.
    def validate_arguments(self):
        if not self.args[0].size[0] == self.args[0].size[1]:
            raise TypeError("The argument '%s' to lambda_min must resolve to a square matrix."
                % self.args[0].name())

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.NONMONOTONIC]

    def graph_implementation(self, arg_objs):
        A = arg_objs[0]
        n,m = A.size
        # Requires that A is symmetric.
        constr = (A == A.T).canonical_form[1]
        # SDP constraint.
        t = Variable()
        I = Constant(np.eye(n,m))
        return (t, [SDP(A - I*t)] + constr)
