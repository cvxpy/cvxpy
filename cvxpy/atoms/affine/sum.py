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

# from ... import utilities as u
# from affine_atom import AffAtom
# import numpy as np
# import operator as op

# class sum(AffAtom):
#     """ Sum of the entries in an expression. """
#     # expr - the expression to be summed.
#     def __init__(self, expr):
#         super(sum, self).__init__(expr)

#     # Sums along both axes.
#     @AffAtom.numpy_numeric
#     def numeric(self, values):
#         return np.sum(np.sum(values[0]))

#     # The shape, sign, and curvature of the index/slice.
#     def _dcp_attr(self):
#         arg_attr = self.args[0]._dcp_attr()
#         index_attr = []
#         for row in arg_attr.shape.size[0]:
#             for col in arg_attr.shape.size[1]:
#                 index_attr.append( arg_attr[row,col] )
#         return reduce(op.add, index_attr)

#     # Indexes/slices into the coefficients of the argument.
#     def graph_implementation(self, arg_objs):
#         lh_ones = self.size[0]*[[1]]
#         rh_ones = self.size[1]*[1]
#         return (lh_ones*arg_objs[0]*rh_ones).canonicalize()