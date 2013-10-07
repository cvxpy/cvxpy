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

from ... import settings as s
from ... import utilities as u
from ... import interface as intf
from .. import expression
from .. import leaf
from .. import types

class Variable(leaf.Leaf):
    """ The base variable class """
    VAR_COUNT = 0        
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self._context = u.Context(u.Sign.UNKNOWN, u.Curvature.AFFINE, u.Shape(rows, cols))
        self._init_id()
        self._name = self.id if name is None else name
        self.interface = intf.DEFAULT_INTERFACE
        self.primal_value = None
        super(Variable, self).__init__()

    # Initialize the id.
    def _init_id(self):
        self.id = Variable.next_var_name()

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        Variable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self._name

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Returns the variable object mapped to an identity matrix.
    def coefficients(self, interface):
        # Scalars have a scalar coefficient.
        if self.size == (1,1):
            coeff = 1
        else:
            coeff = interface.identity(self.size[0])
        return {self: coeff}

    # Return self.
    def variables(self):
        return [self]

    # Return a view into a matrix variable.
    def index_object(self, key):
        return types.index_variable()(self, key)

    # The transpose of the variable.
    def transpose(self):
        return types.transpose_variable()(self)

    # Adds the coefficient to the matrix for each column in the variable.
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
        horiz_offset = var_offsets[self]
        for col in range(self.size[1]):
            interface.block_add(matrix, coeff, vert_offset, horiz_offset, rows, cols)
            horiz_offset += cols
            vert_offset += rows