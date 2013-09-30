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
from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..constraints.affine import AffEqConstraint, AffLeqConstraint
from ..constraints.semi_definite import SDP
from ..interface import numpy_wrapper as np

class lambda_max(Atom):
    """ Maximum eigenvalue. """
    def __init__(self, A):
        super(lambda_max, self).__init__(A)

    # Resolves to a scalar.
    def set_shape(self):
        self.validate_arguments()
        self._shape = u.Shape(1,1)

    # Verify that the argument A is square.
    def validate_arguments(self):
        if not self.args[0].size[0] == self.args[0].size[1]:
            raise TypeError("The argument '%s' to lambda_max must resolve to a square matrix." 
                % self.args[0].name())

    # Always unknown.
    def sign_from_args(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]
    
    # TODO Assumes A is symmetric.
    @staticmethod
    def graph_implementation(var_args, size):
        A = var_args[0]
        t = Variable(*size).canonical_form()[0]
        I = Constant(np.eye(*A.size)).canonical_form()[0]
        return (t, [SDP(I*t - A)] + constr)