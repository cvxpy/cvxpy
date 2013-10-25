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

from affine_atom import AffAtom
from ... import utilities as u
from ...utilities import bool_mat_utils as bu
from ... import interface as intf
from ...expressions import types
from ...expressions.variables import Variable
from ...expressions.affine import AffExpression
from ...expressions.vstack import AffVstack
from collections import deque
import numpy as np

class vstack(AffAtom):
    """ Vertical concatenation """
    # Returns the vstack of the values.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.vstack(values)
        
    # The shape is the common width and the sum of the heights.
    def get_shape(self):
        self.validate_arguments()
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        return u.Shape(rows, cols)


    # All arguments must have the same width.
    def validate_arguments(self):
        arg_cols = [arg.size[1] for arg in self.args]
        if max(arg_cols) != min(arg_cols):
            raise TypeError( ("All arguments to vstack must have "
                              "the same number of columns.") )

    # Vertically concatenates sign and curvature as a dense matrix.
    def get_sign_curv(self):
        sizes = [arg.size for arg in self.args]
        # Sign.
        neg_mat = bu.vstack([arg.sign.neg_mat for arg in self.args], sizes)
        pos_mat = bu.vstack([arg.sign.pos_mat for arg in self.args], sizes)
        # Curvature.
        cvx_mat = bu.vstack([arg.curvature.cvx_mat for arg in self.args], sizes)
        conc_mat = bu.vstack([arg.curvature.conc_mat for arg in self.args], sizes)
        constant = all(arg.curvature.is_constant() for arg in self.args)

        return (u.Sign(neg_mat, pos_mat),
                u.Curvature(cvx_mat, conc_mat, constant))

    # Sets the shape, sign, and curvature.
    def set_context(self):
        shape = self.get_shape()
        sign,curvature = self.get_sign_curv()
        self._context = u.Context(sign, curvature, shape)

    @staticmethod
    def graph_implementation(var_args, size):
        obj = AffVstack(*var_args)
        obj = AffExpression(obj.variables(), [deque([obj])], obj._shape)
        return (obj, [])

    # Return the the component of vstack at the given index.
    # TODO replace with binary tree.
    def index_object(self, key):
        index = 0
        offset = 0
        while offset + self.args[index].size[0] <= key[0].start:
            offset += self.args[index].size[0]
            index += 1
        return self.args[index][key[0].start - offset, key[1].start]