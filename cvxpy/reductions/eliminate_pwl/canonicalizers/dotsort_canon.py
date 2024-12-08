"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

from cvxpy import Constant
from cvxpy.atoms.affine.binary_operators import outer
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.expressions.variable import Variable


def dotsort_canon(expr, args):
    x = args[0]
    w = args[1]

    if isinstance(w, Constant):
        w_unique, w_counts = np.unique(w.value, return_counts=True)
    else:
        w_unique, w_counts = w, np.ones(w.size)  # Can't group by unique elements for parameters

    # minimize    sum(t) + q @ w_counts
    # subject to  x @ w_unique.T <= t + q.T
    #             0 <= t

    t = Variable((x.size, 1), nonneg=True)
    q = Variable((1, w_unique.size))

    obj = sum(t) + q @ w_counts
    x_w_unique_outer_product = outer(vec(x, order='F'), vec(w_unique, order='F'))
    constraints = [x_w_unique_outer_product <= t + q]
    return obj, constraints
