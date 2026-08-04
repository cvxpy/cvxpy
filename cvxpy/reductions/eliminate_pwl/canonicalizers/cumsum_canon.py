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
import scipy.sparse as sp
from numpy.lib.array_utils import normalize_axis_index

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.solver_context import SolverInfo


def cumsum_canon(expr, args, solver_context: SolverInfo | None = None):
    """Cumulative sum."""
    X = args[0]
    axis = expr.axis
    shape = expr.shape

    # Handle axis=None: flatten in C order, then treat as 1D with axis=0
    if axis is None:
        X = reshape(X, (X.size,), order='C')
        shape = (X.size,)
        axis = 0

    ndim = len(shape)
    axis = normalize_axis_index(axis, ndim)

    # If only one element along this axis, cumsum is identity
    if shape[axis] == 1:
        return X, []

    if not X.variables():
        # X is constant or parameter-affine. Introducing an auxiliary Variable
        # here would break DPP: e.g., in cumsum(param) @ x the parameter-affine
        # factor would become a Variable, making the product non-DPP. Instead,
        # apply an explicit lower-triangular ones matrix along the cumsum axis.
        # This dense triangular matrix is avoided for variable arguments (the
        # constraint-based path below), but parameter-only arguments are
        # typically small, so the O(n^2) matrix is an acceptable trade-off.
        dim = shape[axis]
        tril = sp.csc_array(np.tril(np.ones((dim, dim))))
        pre = int(np.prod(shape[:axis], dtype=int))
        post = int(np.prod(shape[axis + 1:], dtype=int))
        # Operator on the F-order vectorization of X: axes before `axis` vary
        # fastest, so cumsum along `axis` is I_post (x) tril (x) I_pre.
        op = sp.kron(sp.eye_array(post), sp.kron(tril, sp.eye_array(pre)))
        flat = reshape(X, (X.size,), order='F')
        Y = reshape(Constant(sp.csc_array(op)) @ flat, shape, order='F')
        return Y, []

    Y = Variable(shape)

    # Build slices for "all but first" and "all but last" along axis
    # and for "first element" along axis
    slice_all = slice(None)
    slice_from_1 = slice(1, None)
    slice_to_minus1 = slice(None, -1)
    slice_first = slice(0, 1)

    # Create index tuples
    idx_from_1 = tuple(slice_from_1 if i == axis else slice_all for i in range(ndim))
    idx_to_minus1 = tuple(slice_to_minus1 if i == axis else slice_all for i in range(ndim))
    idx_first = tuple(slice_first if i == axis else slice_all for i in range(ndim))

    # X[from_1] = Y[from_1] - Y[to_minus1]
    # Y[first] = X[first]
    constr = [X[idx_from_1] == Y[idx_from_1] - Y[idx_to_minus1],
              Y[idx_first] == X[idx_first]]
    return Y, constr
