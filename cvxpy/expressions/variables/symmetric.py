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
from cvxpy.expressions import cvxtypes
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp


def Symmetric(n, name=None):
    """An expression representing a symmetric matrix.
    """
    var = SymmetricUpperTri(n, name)
    fill_mat = Constant(upper_tri_to_full(n))
    return cvxtypes.reshape()(fill_mat*var, int(n), int(n))


def upper_tri_to_full(n):
    """Returns a coefficient matrix to create a symmetric matrix.

    Parameters
    ----------
    n : int
        The width/height of the matrix.

    Returns
    -------
    SciPy CSC matrix
        The coefficient matrix.
    """
    entries = n*(n+1)//2

    val_arr = []
    row_arr = []
    col_arr = []
    count = 0
    for i in range(n):
        for j in range(i, n):
            # Index in the original matrix.
            col_arr.append(count)
            # Index in the filled matrix.
            row_arr.append(j*n + i)
            val_arr.append(1.0)
            if i != j:
                # Index in the original matrix.
                col_arr.append(count)
                # Index in the filled matrix.
                row_arr.append(i*n + j)
                val_arr.append(1.0)
            count += 1

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (n*n, entries)).tocsc()


class SymmetricUpperTri(Variable):
    """ The upper triangular part of a symmetric variable. """

    def __init__(self, n, name=None):
        self.n = n
        super(SymmetricUpperTri, self).__init__(n*(n+1)//2, 1, name)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.n, self.name]

    def canonicalize(self):
        upper_tri = lu.create_var((self.size[0], 1), self.id)
        return (upper_tri, [])

    def __repr__(self):
        """String to recreate the object.
        """
        return "SymmetricUpperTri(%d)" % self.n
