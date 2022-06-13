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

from cvxpy import reshape
from cvxpy.atoms.affine.sum import sum
from cvxpy.expressions.variable import Variable


def dotsort_canon(expr, args):
    x = args[0]
    w = args[1]

    w_unique, w_counts = np.unique(w.value, return_counts=True)

    # minimize    sum(t) + q @ w_counts
    # subject to  x @ w.T <= t + q.T
    #             0 <= t

    t = Variable((x.size, 1), nonneg=True)
    q = Variable((1, len(w_unique)))

    obj = sum(t) + q @ w_counts
    x_w_unique_outer_product = reshape(x, (x.size, 1)) @ w_unique.reshape((1, -1))
    constraints = [x_w_unique_outer_product <= t + q]
    return obj, constraints
