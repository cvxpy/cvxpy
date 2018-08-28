"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms import reshape, trace
from cvxpy.expressions.variable import Variable


def matrix_frac_canon(expr, args):
    X = args[0]  # n by m matrix.
    P = args[1]  # n by n matrix.

    if len(X.shape) == 1:
        X = reshape(X, (X.shape[0], 1))
    n, m = X.shape

    # Create a matrix with Schur complement T - X.T*P^-1*X.
    M = Variable((n+m, n+m), PSD=True)
    T = Variable((m, m), symmetric=True)
    constraints = []
    # Fix M using the fact that P must be affine by the DCP rules.
    # M[0:n, 0:n] == P.
    constraints.append(M[0:n, 0:n] == P)
    # M[0:n, n:n+m] == X
    constraints.append(M[0:n, n:n+m] == X)
    # M[n:n+m, n:n+m] == T
    constraints.append(M[n:n+m, n:n+m] == T)
    return trace(T), constraints
