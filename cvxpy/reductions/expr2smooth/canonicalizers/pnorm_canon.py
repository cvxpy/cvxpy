"""
Copyright 2025 CVXPY developers

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


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.p

    if p == 1:
        return x, []

    shape = expr.shape
    t = Variable(shape)
    if p % 2 == 0:
        summation = sum([x[i]**p for i in range(x.size)])
        return t, [t**p == summation, t >= 0]
