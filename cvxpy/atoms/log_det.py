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

from atom import Atom
from elementwise.log import log
from det_rootn import det_rootn

def log_det(A):
    """Alias for n*log(det_rootn(A))

    Args:
        A: A square matrix.

    Returns:
        A scalar Expression.
    """
    A = Atom.cast_to_const(A)
    rows, cols = A.size
    return rows*log(det_rootn(A))
