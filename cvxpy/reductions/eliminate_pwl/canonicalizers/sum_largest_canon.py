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

from cvxpy.atoms import promote, reshape
from cvxpy.atoms.affine.sum import sum
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def sum_largest_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    k = expr.k
    axis = expr.axis

    # min sum(t, axis) + k*q
    # s.t. x <= t + q
    #      0 <= t
    t = Variable(x.shape)
    q = Variable(expr.shape)

    if axis is None:
        promoted_q = promote(q, x.shape)
    else:
        axes = {axis} if isinstance(axis, int) else set(axis)
        keepdims_shape = tuple(1 if i in axes else s for i, s in enumerate(x.shape))
        promoted_q = reshape(q, keepdims_shape, order='F')

    obj = sum(t, axis=axis, keepdims=expr.keepdims) + k * q
    constraints = [x <= t + promoted_q, t >= 0]
    return obj, constraints
