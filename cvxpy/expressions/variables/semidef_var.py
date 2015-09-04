"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variables.variable import Variable
from cvxpy.expressions.variables.symmetric import upper_tri_to_full
from cvxpy.constraints.semidefinite import SDP
import cvxpy.expressions.types as types
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp

def Semidef(n, name=None):
    """An expression representing a positive semidefinite matrix.
    """
    var = SemidefUpperTri(n, name)
    fill_mat = Constant(upper_tri_to_full(n))
    return types.reshape()(fill_mat*var, n, n)

class SemidefUpperTri(Variable):
    """ The upper triangular part of a positive semidefinite variable. """
    def __init__(self, n, name=None):
        self.n = n
        super(SemidefUpperTri, self).__init__(n*(n+1)//2, 1, name)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.n, self.name]

    def canonicalize(self):
        """Variable must be semidefinite and symmetric.
        """
        upper_tri = lu.create_var((self.size[0], 1), self.id)
        fill_coeff = upper_tri_to_full(self.n)
        fill_coeff = lu.create_const(fill_coeff, (self.n*self.n, self.size[0]),
                                     sparse=True)
        full_mat = lu.mul_expr(fill_coeff, upper_tri, (self.n*self.n, 1))
        full_mat = lu.reshape(full_mat, (self.n, self.n))
        return (upper_tri, [SDP(full_mat, enforce_sym=False)])

    def __repr__(self):
        """String to recreate the object.
        """
        return "SemidefUpperTri(%d)" % self.n
