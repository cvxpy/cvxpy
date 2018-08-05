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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
import scipy.sparse as sp


def sigma_max_canon(expr, args):
    A = args[0]
    n, m = A.shape
    X = Variable((n+m, n+m), PSD=True)

    shape = expr.shape
    t = Variable(shape)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:n, 0:n] == I_n*t
    constraints.append(X[0:n, 0:n] == Constant(sp.eye(n)) * t)

    # X[0:n, n:n+m] == A
    constraints.append(X[0:n, n:n+m] == A)

    # X[n:n+m, n:n+m] == I_m*t
    constraints.append(X[n:n+m, n:n+m] == Constant(sp.eye(m)) * t)

    return t, constraints
