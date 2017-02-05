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
