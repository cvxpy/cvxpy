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
from ...constraints.affine import AffEqConstraint

class transpose(AffAtom):
    """ Matrix transpose. """
    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0].T

    # Transposes shape, sign, and curvature.
    def set_context(self):
        rows,cols = self.args[0].size
        shape = u.Shape(cols, rows)

        neg_mat = bu.transpose(self.args[0].sign.neg_mat)
        pos_mat = bu.transpose(self.args[0].sign.pos_mat)
        cvx_mat = bu.transpose(self.args[0].curvature.cvx_mat)
        conc_mat = bu.transpose(self.args[0].curvature.conc_mat)
        constant = self.args[0].curvature.constant

        self._context = u.Context(u.Sign(neg_mat, pos_mat),
                                  u.Curvature(cvx_mat, conc_mat, constant), 
                                  shape)

    # Create a new variable equal to the argument transposed.
    @staticmethod
    def graph_implementation(var_args, size):
        X = Variable(size[1], size[0])
        obj = X.T.canonical_form()[0]
        return (obj, [AffEqConstraint(X, var_args[0])])

    # Index the original argument as if it were the transpose,
    # then return the transpose.
    def index_object(self, key):
        return transpose(self.args[0][key[1], key[0]])