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
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.values import make_canon_variable


def max_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    axis = expr.axis
    t = make_canon_variable(expr, solver_context)

    if axis is None:
        promoted_t = promote(t, x.shape)
    else:
        axes = {axis} if isinstance(axis, int) else set(axis)
        keepdims_shape = tuple(1 if i in axes else s for i, s in enumerate(x.shape))
        promoted_t = reshape(t, keepdims_shape, order='F')

    constraints = [x <= promoted_t]

    return t, constraints
