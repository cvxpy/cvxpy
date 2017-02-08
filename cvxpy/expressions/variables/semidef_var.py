"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variables.variable import Variable
from cvxpy.expressions.variables.symmetric import upper_tri_to_full
from cvxpy.constraints.semidefinite import SDP
from cvxpy.expressions import cvxtypes
import cvxpy.lin_ops.lin_utils as lu


def Semidef(n, name=None):
    """An expression representing a positive semidefinite matrix.
    """
    var = SemidefUpperTri(n, name)
    fill_mat = Constant(upper_tri_to_full(n))
    return cvxtypes.reshape()(fill_mat*var, n, n)


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
