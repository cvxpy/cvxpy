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

from typing import List, Tuple

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable


def normNuc_canon(expr, args) -> Tuple[float, List[Constraint]]:
    A = args[0]
    m, n = A.shape
    # Create the equivalent problem:
    #   minimize (trace(U) + trace(V))/2
    #   subject to:
    #            [U A; A.T V] is positive semidefinite
    constraints = []
    U = Variable(shape=(m, m), symmetric=True)
    V = Variable(shape=(n, n), symmetric=True)
    X = bmat([[U, A],
              [A.T, V]])
    constraints.append(X >> 0)
    trace_value = 0.5 * (trace(U) + trace(V))
    return trace_value, constraints
