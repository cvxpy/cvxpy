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

import numpy as np

from cvxpy.atoms.affine.sum import sum
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def sum_largest_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    k = expr.k

    # min sum(t) + kq
    # s.t. x <= t + q
    #      0 <= t
    t = Variable(x.shape)
    q = Variable()
    obj = sum(t) + k*q
    constraints = [x <= t + q, t >= 0]

    # for DNLP we must initialize the new variable (DNLP guarantees that 
    # x.value will be set when this function is called). The initialization 
    # below is motivated by the optimal solution of min sum(t) + kq subject
    # to x <= t + q, 0 <= t
    if x.value is not None:
        sorted_indices = np.argsort(x.value)
        idx_of_smallest = sorted_indices[:k]
        idx_of_largest = sorted_indices[-k:]
        q.value = np.max(x.value[idx_of_smallest])
        t.value = np.zeros_like(x.value)
        t.value[idx_of_largest] = x.value[idx_of_largest] - q.value
        
    return obj, constraints
