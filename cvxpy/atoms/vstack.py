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
from cvxpy.expressions.affine import AffObjective
from cvxpy.expressions.vstack import AffVstack
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
from collections import deque

class vstack(Atom):
    """ Vertical concatenation """
    # The shape is the common width and the sum of the heights.
    def set_shape(self):
        self.validate_arguments()
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        self._shape = u.Shape(rows, cols)

    @property
    def sign(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.AFFINE

    def monotonicity(self): # TODO what would make sense?
        return len(self.args)*[u.Monotonicity.INCREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        arg_cols = [arg.size[1] for arg in self.args]
        if max(arg_cols) != min(arg_cols):
            raise TypeError( ("All arguments to vstack must have "
                              "the same number of columns.") )

    @staticmethod
    def graph_implementation(var_args, size):
        obj = AffVstack(*var_args)
        obj = AffObjective(obj.variables(), [deque([obj])], obj._shape)
        return (obj, [])

    # Return the absolute value of the argument at the given index.
    # TODO replace with binary tree.
    def index_object(self, key):
        index = 0
        offset = 0
        while offset + self.args[index].size[0] <= key[0]:
            offset += self.args[index].size[0]
            index += 1
        return self.args[index][key[0] - offset, key[1]]