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

from cvxpy.atoms import bmat, reshape, trace, upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable


def matrix_frac_canon(expr, args):
    X = args[0]  # n by m matrix.
    P = args[1]  # n by n matrix.

    if len(X.shape) == 1:
        X = reshape(X, (X.shape[0], 1))
    n, m = X.shape
    T = Variable((m, m), symmetric=True)
    M = bmat([[P, X],
              [X.T, T]])
    # ^ a matrix with Schur complement T - X.T*P^-1*X.
    constraints = [PSD(M)]
    if not P.is_symmetric():
        ut = upper_tri(P)
        lt = upper_tri(P.T)
        constraints.append(ut == lt)
    return trace(T), constraints
