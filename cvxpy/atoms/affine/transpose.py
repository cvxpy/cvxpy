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
from ...expressions.variables import Variable
from ...expressions.affine import AffExpression

class transpose(AffAtom):
    """ Matrix transpose. """
    # The string representation of the atom.
    def name(self):
        return "%s.T" % self.args[0]

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0].T

    # Transposes shape, sign, and curvature.
    def _dcp_attr(self):
        rows,cols = self.args[0].size
        shape = u.Shape(cols, rows)

        neg_mat = bu.transpose(self.args[0].sign.neg_mat)
        pos_mat = bu.transpose(self.args[0].sign.pos_mat)
        cvx_mat = bu.transpose(self.args[0].curvature.cvx_mat)
        conc_mat = bu.transpose(self.args[0].curvature.conc_mat)
        constant = self.args[0].curvature.constant

        return u.DCPAttr(u.Sign(neg_mat, pos_mat),
                         u.Curvature(cvx_mat, conc_mat, constant), 
                         shape)

    # Create a new variable equal to the argument transposed.
    def graph_implementation(self, arg_objs):
        X = Variable(self.size[1], self.size[0])
        # Create a coefficients dict for the transposed variable.
        # Each row in each block selects the appropriate elements
        # from the vectorized X.
        var_blocks = X.coefficients()[X]
        transpose_coeffs = X.init_coefficients(self.size[0], self.size[1])
        transpose_blocks = transpose_coeffs[X]
        for k in xrange(self.size[0]):
            for row in xrange(self.size[1]):
                transpose_blocks[row][k,:] = var_blocks[k][row,:]
        transpose_coeffs[X] = transpose_blocks
        
        obj = AffExpression(transpose_coeffs, X._variables, self._dcp_attr())
        return (obj, [X == arg_objs[0]])