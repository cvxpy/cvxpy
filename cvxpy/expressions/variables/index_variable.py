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

from ... import utilities as u
from variable import Variable

class IndexVariable(Variable):
    """ An index into a matrix variable """
    # parent - the variable indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        name = parent.name() + "[%s,%s]" % u.Key.to_str(self.key)
        super(IndexVariable, self).__init__(name=name)

    # Return parent so that the parent value is updated.
    def variables(self):
        return [self.parent]

    # Initialize the id.
    def _init_id(self):
        self.id = self.parent.index_id(*self.key)

    # Convey the parent's constraints to the canonicalization.
    def _constraints(self):
        return self.parent._constraints()

    # The value at the index.
    @property
    def value(self):
        if self.parent.value is None:
            return None
        else:
            return self.parent.value[self.key]

    # Vectorizes the coefficient and adds it to the matrix.
    # matrix - the coefficient matrix.
    # coeff - the coefficient for the variable.
    # vert_offset - the current vertical offset.
    # constraint - the constraint containing the variable. 
    # var_offsets - a map of variable object to horizontal offset.
    # interface - the interface for the matrix type.
    def place_coeff(self, matrix, coeff, vert_offset, 
                    constraint, var_offsets, interface):
        rows = constraint.size[0]*constraint.size[1]
        horiz_offset = var_offsets[self.parent]
        horiz_offset += self.key[0].start + self.key[1].start*self.parent.size[0]
        interface.block_add(matrix, coeff, vert_offset, horiz_offset, rows, 1)