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

from cvxpy.atoms.affine.trace import trace
from cvxpy.expressions.variable import Variable


def normNuc_canon(expr, args):
    A = args[0]
    m, n = A.shape

    # Create the equivalent problem:
    #   minimize (trace(U) + trace(V))/2
    #   subject to:
    #            [U A; A.T V] is positive semidefinite
    X = Variable((m+n, m+n), PSD=True)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:rows,rows:rows+cols] == A
    constraints.append(X[0:m, m:m+n] == A)
    trace_value = 0.5 * trace(X)

    return trace_value, constraints
