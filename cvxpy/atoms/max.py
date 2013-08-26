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
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class max(Atom):
    """ Elementwise maximum. """
    # The shape is the common shape of all the arguments.
    def set_shape(self):
        shape = self.args[0]._shape
        for arg in self.args[1:]:
            shape = shape + arg._shape
        self._shape = shape

    """
    Reduces the list of argument signs according to the following rules:
        POSITIVE, ANYTHING = POSITIVE
        ZERO, UNKNOWN = POSITIVE
        ZERO, ZERO = ZERO
        ZERO, NEGATIVE = ZERO
        UNKNOWN, NEGATIVE = UNKNOWN
        NEGATIVE, NEGATIVE = NEGATIVE
    """
    def sign_from_args(self):
        neg_mat = self.args[0].sign.neg_mat
        pos_mat = self.args[0].sign.pos_mat
        for arg in self.args[1:]:
            neg_mat = neg_mat & arg.sign.neg_mat
            pos_mat = pos_mat | arg.sign.pos_mat
        return u.Sign(neg_mat, pos_mat)

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return len(self.args)*[u.Monotonicity.INCREASING]
    
    @staticmethod
    def graph_implementation(var_args, size):
        t = Variable(*size)
        constraints = [AffLeqConstraint(x, t) for x in var_args]
        return (t, constraints)

    # Return the max of the arguments' elements at the given index.
    def index_object(self, key):
        args = []
        for arg in self.args:
            if arg.size == (1,1):
                args.append(arg)
            else:
                args.append(arg[key])
        return self.__class__(*args)