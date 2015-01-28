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

from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variables import Semidef
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.affine.trace import trace

def lambda_sum_largest(X, k):
    """Sum of the largest k eigenvalues.
    """
    X = Expression.cast_to_const(X)
    if X.size[0] != X.size[1]:
        raise ValueError("First argument must be a square matrix.")
    elif int(k) != k or k <= 0:
        raise ValueError("Second argument must be a positive integer.")
    """
    S_k(X) denotes lambda_sum_largest(X, k)
    t >= k S_k(X - Z) + trace(Z), Z is PSD
    implies
    t >= ks + trace(Z)
    Z is PSD
    sI >= X - Z (PSD sense)
    which implies
    t >= ks + trace(Z) >= S_k(sI + Z) >= S_k(X)
    We use the fact that
    S_k(X) = sup_{sets of k orthonormal vectors u_i}\sum_{i}u_i^T X u_i
    and if Z >= X in PSD sense then
    \sum_{i}u_i^T Z u_i >= \sum_{i}u_i^T X u_i

    We have equality when s = lambda_k and Z diagonal
    with Z_{ii} = (lambda_i - lambda_k)_+
    """
    Z = Semidef(X.size[0])
    return k*lambda_max(X - Z) + trace(Z)
