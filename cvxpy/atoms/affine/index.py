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

class index(AffAtom):
    """ Indexing/slicing into a matrix. """
    # expr - the expression indexed/sliced into.
    # key - the index/slicing key (i.e. expr[key[0],key[1]]).
    def __init__(self, expr, key):
        self.key = u.Key.validate_key(key, expr)
        super(index, self).__init__(expr)

    # The string representation of the atom.
    def name(self):
        return self.args[0].name() + "[%s,%s]" % u.Key.to_str(self.key)

    # Returns the index/slice into the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return values[0][self.key]

    # The shape, sign, and curvature of the index/slice.
    def _dcp_attr(self):
        shape = u.Shape(*u.Key.size(self.key, self.args[0]))

        neg_mat = bu.index(self.args[0].sign.neg_mat, self.key)
        pos_mat = bu.index(self.args[0].sign.pos_mat, self.key)
        cvx_mat = bu.index(self.args[0].curvature.cvx_mat, self.key)
        conc_mat = bu.index(self.args[0].curvature.conc_mat, self.key)
        constant = self.args[0].curvature.constant

        return u.DCPAttr(u.Sign(neg_mat, pos_mat),
                         u.Curvature(cvx_mat, conc_mat, constant), 
                         shape)
    
    # Indexes/slices into the coefficients of the argument.    
    def graph_implementation(self, arg_objs):
        used_cols = u.Key.slice_to_set(self.key[1], self.args[0].size[1])
        new_coeffs = {}
        for var,blocks in arg_objs[0].coefficients().items():
            new_blocks = []
            # Indexes into the rows of the coefficients.
            for i,block in enumerate(blocks):
                new_blocks.append( block[self.key[0],:] )
                # Zeros out blocks for unselected columns.
                if i not in used_cols:
                    new_blocks[i] *= 0
            new_coeffs[var] = new_blocks

        obj = AffExpression(new_coeffs, arg_objs[0]._variables, self._dcp_attr())
        return (obj, [])