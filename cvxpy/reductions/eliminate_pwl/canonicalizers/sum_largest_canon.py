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

from cvxpy.atoms.affine.sum import sum
from cvxpy.expressions.variable import Variable


def sum_largest_canon(expr, args):
    x = args[0]
    k = expr.k

    # min sum(t) + kq
    # s.t. x <= t + q
    #      0 <= t
    t = Variable(x.shape)
    q = Variable()
    obj = sum(t) + k*q
    constraints = [x <= t + q, t >= 0]
    return obj, constraints
