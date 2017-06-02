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

from scipy.sparse.base import spmatrix
from cvxpy.expressions import expression as exp

BIN_OPS = ["__div__", "__mul__", "__add__", "__sub__",
           "__le__", "__eq__", "__lt__", "__gt__"]


def wrap_bin_op(method):
    """Factory for wrapping binary operators.
    """
    def new_method(self, other):
        if isinstance(other, exp.Expression):
            return NotImplemented
        else:
            return method(self, other)
    return new_method


for method_name in BIN_OPS:
    method = getattr(spmatrix, method_name)
    new_method = wrap_bin_op(method)
    setattr(spmatrix, method_name, new_method)
