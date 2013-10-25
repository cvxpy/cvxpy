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

from ... import interface as intf
from ... import utilities as u
from variable import Variable
import transpose_variable as tv

class IndexVariable(Variable):
    """ An index into a matrix variable """
    # parent - the variable indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        name = parent.name() + "[%s,%s]" % u.Key.to_str(self.key)
        rows,cols = u.Key.size(self.key, self.parent)
        super(IndexVariable, self).__init__(rows, cols, name)

    # Return parent so that the parent value is updated.
    def variables(self):
        return [self.parent]

    # Initialize the id.
    def _init_id(self):
        self.id = self.parent.id + "[%s,%s]" % u.Key.to_str(self.key)

    # Convey the parent's constraints to the canonicalization.
    def _constraints(self):
        return self.parent._constraints()

    # Return a view into the parent matrix variable.
    def index_object(self, key):
        key = u.Key.compose_keys(key, self.key)
        return IndexVariable(self.parent, key)

    # # Transpose the parent and index it.
    # @property
    # def T(self):
    #     return tv.TransposeVariable(self.parent)[self.key[1], self.key[0]]

    # The value at the index.
    @property
    def value(self):
        if self.parent.value is None:
            return None
        else:
            return self.parent.value[self.key]

    # Slices into the coefficient and adds the values to 
    # the appropriate slice of the overall matrix.
    # matrix - the coefficient matrix.
    # coeff - the coefficient for the variable.
    # vert_offset - the current vertical offset.
    # constraint - the constraint containing the variable. 
    # var_offsets - a map of variable object to horizontal offset.
    # interface - the interface for the matrix type.
    def place_coeff(self, matrix, coeff, vert_offset, 
                    constraint, var_offsets, interface):
        # Vectorize the coefficient if the variable was promoted.
        if self.size == (1,1):
            rows = constraint.size[0]*constraint.size[1]
        else:
            rows = constraint.size[0]
        cols = self.size[0]
        horiz_offset = var_offsets[self.parent]
        curr_col = self.key[1].start
        while curr_col < u.Key.get_stop(self.key[1], self.parent.size[1]):
            index_offset = curr_col*self.parent.size[0] + self.key[0].start
            interface.block_add(matrix, coeff,
                                vert_offset, horiz_offset + index_offset,
                                rows, cols, horiz_step=self.key[0].step)
            curr_col += self.key[1].step
            vert_offset += rows