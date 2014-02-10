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

from .. import interface as intf
from ..expressions.expression import Expression
from ..expressions.constants import Constant
from norm2 import norm2
from elementwise.square import square
from scipy import linalg as LA
import numpy as np

""" Alias for x.T*P*x """
def quad_form(x, P):
    x,P = map(Expression.cast_to_const, (x,P))
    # Check dimensions.
    n = P.size[0]
    if P.size[1] != n or x.size != (n,1):
        raise Exception("Invalid dimensions for arguments.")
    if x.is_constant():
        return x.T*P*x
    elif P.is_constant():
        np_intf = intf.get_matrix_interface(np.ndarray)
        P = np_intf.const_to_matrix(P.value)
        # Replace P with symmetric version.
        P = (P + P.T)/2
        # Check if P is PSD.
        eigvals, V = LA.eigh(P)
        if min(eigvals) >= 0:
            diag_eig = np.diag(np.sqrt(eigvals))
            P_sqrt = Constant(diag_eig.dot(V.T))
            return square(norm2(P_sqrt*x))
        elif max(eigvals) <= 0:
            diag_eig = np.diag(np.sqrt(-eigvals))
            P_sqrt = Constant(diag_eig.dot(V.T))
            return -square(norm2(P_sqrt*x))
        else:
            raise Exception("P has both positive and negative eigenvalues.")
    else:
        raise Exception("At least one argument to quad_form must be constant.")
