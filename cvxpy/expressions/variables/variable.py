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
from ..affine import AffExpression
from ..constants import Constant
from ..leaf import Leaf

class Variable(Leaf, AffExpression):
    """ The base variable class """
    VAR_COUNT = 0
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self._init_id()
        self._name = self.id if name is None else name
        self.primal_value = None
        dcp_attr = u.DCPAttr(u.Sign.UNKNOWN, 
                             u.Curvature.AFFINE, 
                             u.Shape(rows, cols))
        coeffs = self.init_coefficients(rows, cols)
        variables = {self.id: self}
        super(Variable, self).__init__(coeffs, variables, dcp_attr)

    # Initialize the id.
    def _init_id(self):
        self.id = self.next_name(s.VAR_PREFIX)

    def name(self):
        return self._name

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Returns a coefficients dict with the variable as the key
    # and a list of offset identity matrices as the coefficients.
    def init_coefficients(self, rows, cols):
        # Scalars have scalar coefficients.
        if (rows, cols) == (1,1):
            return {self: [1]}
        else:
            identity = intf.DEFAULT_SPARSE_INTERFACE.identity(rows*cols)
            blocks = [identity[i*rows:(i+1)*rows,:] for i in range(cols)]
            return {self: blocks}

    # # Return self.
    # def variables(self):
    #     return [self]

    # # Return a view into a matrix variable.
    # def index_object(self, key):
    #     return types.index_variable()(self, key)

    # # The transpose of the variable.
    # @property
    # def T(self):
    #     if self.size == (1,1):
    #         return self
    #     else:
    #         return types.transpose_variable()(self)

    # # Adds the coefficient to the matrix for each column in the variable.
    # # matrix - the coefficient matrix.
    # # coeff - the coefficient for the variable.
    # # vert_offset - the current vertical offset.
    # # constraint - the constraint containing the variable.
    # # var_offsets - a map of variable object to horizontal offset.
    # # interface - the interface for the matrix type.
    # def place_coeff(self, matrix, coeff, vert_offset, 
    #                 constraint, var_offsets, interface):
    #     # Vectorize the coefficient if the variable was promoted.
    #     if self.size == (1,1):
    #         rows = constraint.size[0]*constraint.size[1]
    #     else:
    #         rows = constraint.size[0]
    #     cols = self.size[0]
    #     horiz_offset = var_offsets[self]
    #     for col in range(self.size[1]):
    #         interface.block_add(matrix, coeff, vert_offset, horiz_offset, rows, cols)
    #         horiz_offset += cols
    #         vert_offset += rows