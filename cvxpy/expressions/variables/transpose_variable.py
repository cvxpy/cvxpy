# """
# Copyright 2013 Steven Diamond

# This file is part of CVXPY.

# CVXPY is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CVXPY is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
# """

# from variable import Variable
# import index_variable as iv

# class TransposeVariable(Variable):
#     """ The transpose of a matrix variable """
#     # parent - the variable indexed into.
#     def __init__(self, parent):
#         self.parent = parent
#         name = "%s.T" % self.parent.name()
#         rows = self.parent.size[1]
#         cols = self.parent.size[0]
#         super(TransposeVariable, self).__init__(rows, cols, name)

#     # Return parent so that the parent value is updated.
#     def variables(self):
#         return [self.parent]

#     # Initialize the id.
#     def _init_id(self):
#         self.id = "%s.T" % self.parent.id

#     # Convey the parent's constraints to the canonicalization.
#     def _constraints(self):
#         return self.parent._constraints()

#     # The value at the index.
#     @property
#     def value(self):
#         if self.parent.value is None:
#             return None
#         else:
#             return self.parent.value.T

#     # Return a scalar view into a matrix variable.
#     def index_object(self, key):
#         return iv.IndexVariable(self.parent, (key[1],key[0]))

#     # The transpose of the transpose is the original variable.
#     @property
#     def T(self):
#         return self.parent

#     # Splits the coefficient into columns to match the column-major order of
#     # the parent variable in the vector of all variables.
#     # matrix - the coefficient matrix.
#     # coeff - the coefficient for the variable.
#     # vert_offset - the current vertical offset.
#     # constraint - the constraint containing the variable. 
#     # var_offsets - a map of variable object to horizontal offset.
#     # interface - the interface for the matrix type.
#     def place_coeff(self, matrix, coeff, vert_offset, 
#                     constraint, var_offsets, interface):
#         rows = constraint.size[0]
#         stride = self.parent.size[0]
#         for col in range(self.size[1]):
#             for step in range(self.size[0]):
#                 coeff_col = coeff[:,step]
#                 horiz_offset = var_offsets[self.parent] + col + stride*step
#                 interface.block_add(matrix, coeff_col, vert_offset, horiz_offset,
#                                     rows, 1)
#             vert_offset += rows