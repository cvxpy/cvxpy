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

from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def cummax_canon(expr, args, solver_context: SolverInfo | None = None):
    """Cumulative max."""
    X = args[0]
    axis = expr.axis
    ndim = len(expr.shape)

    if expr.shape[axis] == 1:
        return X, []

    # Implicit O(n) definition:
    # Y_{k} = maximum(Y_{k-1}, X_k)
    Y = Variable(expr.shape)
    constr = [X <= Y]

    # Build slices for "all but last" and "all but first" along axis
    slice_prev = tuple(
        slice(None, -1) if i == axis else slice(None) for i in range(ndim)
    )
    slice_next = tuple(
        slice(1, None) if i == axis else slice(None) for i in range(ndim)
    )
    constr += [Y[slice_prev] <= Y[slice_next]]
    return Y, constr
