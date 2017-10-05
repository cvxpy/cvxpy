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

from cvxpy.atoms import trace
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_max_canon import lambda_max_canon


def lambda_sum_largest_canon(expr, args):
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
    X = expr.args[0]
    k = expr.k
    Z = Variable((X.shape[0], X.shape[0]), PSD=True)
    obj, constr = lambda_max_canon(expr, [X - Z])
    obj = k*obj + trace(Z)
    return obj, constr
