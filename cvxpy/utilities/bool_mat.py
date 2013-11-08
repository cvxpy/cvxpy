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
# import numpy as np

# class BoolMat(object):
#     """ 
#     A wrapper on a boolean numpy ndarray for use as a matrix
#     to hold signs and curvatures.
#     """
#     ND_TRUE_SCALAR = np.atleast_2d(True)
#     # Initializes the BoolMat value, reduced to a scalar if possible.
#     # value - the underlying ndarray.
#     def __init__(self, value):
#         if value.all():
#             value = ND_TRUE_SCALAR
#         elif not value.any():
#             value = ~ND_TRUE_SCALAR
#         self.value = value

#     # Promotes a scalar BoolMat to a BoolMat of the given dimensions.
#     @staticmethod
#     def promote(value, size):
#         mat = np.empty(size, dtype='bool')
#         mat.fill(value)
#         return BoolMat(mat)

#     # Return True if any entry is True.
#     def any(self):
#         return self.value.any()

#     # Index/slice into the BoolMat.
#     def __getitem__(self, key):
#         return BoolMat(self.value[key])

#     # For addition.
#     def __or__(self, other):
#         if isinstance(other, BoolMat):
#             return BoolMat(self.value | other.value)
#         else:
#             return NotImplemented

#     # Handles boolean | BoolMat
#     def __ror__(self, other):
#         return self | other

#     # Handles multiplication with SparseBoolMat.
#     def __mul__(self, other):
#         if isinstance(other, BoolMat):
#             mult_val = self.value.dot(other.value)
#             return BoolMat(mult_val)
#         else:
#             return NotImplemented

#     # For elementwise multiplication/bitwise and.
#     def __and__(self, other):
#         if isinstance(other, BoolMat):
#             mult_val = self.value & other.value
#             return BoolMat(mult_val)
#         else:
#             return NotImplemented

#     # Handles boolean & BoolMat
#     def __rand__(self, other):
#         return self & other

#     # For comparison.
#     def __eq__(self, other):
#         if isinstance(other, BoolMat):
#             return self.value.shape == other.value.shape and \
#                    (self.value == other.value).all()
#         else:
#             return NotImplemented

#     # To string methods.
#     def __repr__(self):
#         return str(self.value)

# # Scalar BoolMats for use by Curvature and Sign.
# TRUE_MAT = BoolMat(BoolMat.ND_TRUE_SCALAR)
# FALSE_MAT = BoolMat(~BoolMat.ND_TRUE_SCALAR)